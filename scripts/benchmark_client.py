"""Evaluate a served policy in LIBERO, LIBERO-Plus, or SimplerEnv."""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import math
import os
import pickle
import re
import runpy
import sys
import tempfile
import zipfile
from collections import deque
from contextlib import ExitStack
from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from websockets.sync.client import connect


def pack_array(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def unpack_array(payload: dict) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    shape = tuple(int(size) for size in payload["shape"])
    raw = base64.b64decode(payload["data"], validate=True)
    if len(raw) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("server returned an invalid action array")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


class PolicyClient:
    """Small, synchronous client for the policy server's JSON WebSocket RPC."""

    def __init__(self, url: str, *, timeout: float = 600.0) -> None:
        self.timeout = float(timeout)
        options = {"max_size": 64 * 1024 * 1024}
        if "proxy" in inspect.signature(connect).parameters:
            options["proxy"] = None
        self._connection_context = connect(url, **options)
        self.connection = self._connection_context.__enter__()
        self.next_id = 0
        self.metadata = self.call("metadata")
        self.plan: deque[np.ndarray] = deque()
        self.previous_action: np.ndarray | None = None

    def close(self) -> None:
        self._connection_context.__exit__(None, None, None)

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        self.connection.send(json.dumps({
            "id": self.next_id, "method": method, "params": params or {},
        }, separators=(",", ":")))
        response = json.loads(self.connection.recv(timeout=self.timeout))
        if response.get("id") != self.next_id:
            raise RuntimeError("policy server returned the wrong request id")
        if "error" in response:
            raise RuntimeError(f"policy server: {response['error']}")
        return response["result"]

    def reset(self) -> None:
        self.plan.clear()
        spec = self.metadata["action_spec"]
        self.previous_action = np.zeros(int(spec["dimension"]), dtype=np.float32)

    def next_action(self, frames: list[np.ndarray], task: str, state: np.ndarray | None) -> np.ndarray:
        if not self.plan:
            params = {
                "frames": [pack_array(np.asarray(frame, dtype=np.uint8)) for frame in frames],
                "task": task,
            }
            if state is not None:
                params["state"] = np.asarray(state, dtype=np.float32).tolist()
            if self.metadata["requires_previous_action"]:
                params["previous_action"] = self.previous_action.tolist()
            result = self.call("predict", params)
            actions = unpack_array(result["actions"])
            if actions.ndim != 2 or actions.shape[1] != self.metadata["action_spec"]["dimension"]:
                raise ValueError(f"policy returned an invalid action shape: {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("policy returned non-finite actions")
            stride = int(self.metadata["execution_horizon"])
            if not 1 <= stride <= len(actions):
                raise ValueError("policy execution_horizon exceeds its action chunk")
            self.plan.extend(actions[:stride])
        action = self.plan.popleft()
        self.previous_action = action.copy()
        return action


def gripper_is_open(raw: float, spec: dict) -> bool:
    """Whether a policy gripper value means open, under the checkpoint's own convention."""
    open_is_high = spec["gripper_open"] > spec["gripper_closed"]
    return (raw > spec["gripper_binary_threshold"]) == open_is_high


def require_contract(metadata: dict, *, benchmark: str) -> None:
    spec = metadata["action_spec"]
    required = {"space": "delta_eef", "dimension": 7, "frame": "robot_base"}
    if benchmark == "simplerenv":
        required.update(rotation_representation="euler_xyz", control_frequency_hz=5.0)
        if metadata["proprio_mode"] not in {"none", "zero"}:
            raise ValueError("SimplerEnv WidowX client needs a vision-only policy")
        if metadata.get("camera_keys") and len(metadata["camera_keys"]) != 1:
            raise ValueError("SimplerEnv WidowX client needs a one-camera policy")
    else:
        required.update(rotation_representation="rotation_vector")
        if not spec.get("control_frequency_hz"):
            raise ValueError(
                "the checkpoint does not state its control_frequency_hz, so the LIBERO "
                "environment rate cannot be matched to it"
            )
        if metadata.get("camera_keys") and len(metadata["camera_keys"]) != 2:
            raise ValueError("LIBERO client needs a two-camera policy")
    mismatches = {name: (spec.get(name), value) for name, value in required.items()
                  if spec.get(name) != value}
    if mismatches:
        raise ValueError(f"checkpoint action contract does not fit {benchmark}: {mismatches}")


def _libero_task_names(repo: Path, suite: str) -> list[str]:
    task_map = runpy.run_path(str(
        repo / "libero/libero/benchmark/libero_suite_task_map.py"
    ))["libero_task_map"]
    if suite not in task_map:
        raise ValueError(f"unknown LIBERO suite {suite!r}; available: {sorted(task_map)}")
    return list(task_map[suite])


def _libero_init_states(path: Path, *, single_state: bool = False) -> np.ndarray:
    """Read trusted LIBERO NumPy init states from the torch zip without torch."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError(f"unexpected LIBERO init archive: {path}")
        states = np.asarray(pickle.loads(archive.read(names[0])), dtype=np.float32)
    if single_state and states.ndim == 1:
        states = states.reshape(1, -1)
    if states.ndim != 2 or not np.isfinite(states).all():
        raise ValueError(f"invalid LIBERO init states: {path}: {states.shape}")
    return states


def _task_language(name: str) -> str:
    if name[0].isupper():
        offset = 8 if "SCENE10" in name else 7
        return " ".join(name[name.find("SCENE") + offset:].split("_"))
    return " ".join(name.split("_"))


@cache
def _plus_classifications(repo: Path, suite: str) -> dict[str, str]:
    path = repo / "libero/libero/benchmark/task_classification.json"
    rows = json.loads(path.read_text(encoding="utf-8"))[suite]
    return {row["name"]: row["category"] for row in rows}


def _plus_task_info(repo: Path, suite: str, name: str, bddl_root: Path,
                    init_root: Path) -> tuple[str, Path, bool, bool]:
    category = _plus_classifications(repo, suite)[name]
    if category == "Language Instructions":
        bddl_name = name.split("_view_", 1)[0]
        lines = (bddl_root / suite / f"{bddl_name}.bddl").read_text().splitlines()
        language = next(" ".join(line.strip()[11:-1].lower().split())
                        for line in lines if line.strip().startswith("(:language "))
    else:
        suffixes = (r"_view_", r"_table_\d+(?:_|$)", r"_tb_\d+(?:_|$)",
                    r"_light_\d+(?:_|$)", r"_add_", r"_moved_level", r"_level")
        cuts = [match.start() for pattern in suffixes
                if (match := re.search(pattern, name)) is not None]
        language = _task_language(name[:min(cuts)] if cuts else name)
    if "_language_" in name:
        base = name.split("_language_", 1)[0]
    elif "_view_" in name:
        base = name.split("_view_", 1)[0]
    else:
        base = re.sub(r"_tb_\d+", "", re.sub(r"_table_\d+", "", name))
        if "_light_" in base:
            base = base.split("_light_", 1)[0]
        if "_add_" in name or "_level" in name:
            return language, init_root / "libero_newobj" / suite / f"{name}.pruned_init", True, category == "Robot Initial States"
    return language, init_root / suite / f"{base}.pruned_init", False, category == "Robot Initial States"


def _with_robot_qpos(env, initial_state: np.ndarray) -> np.ndarray:
    state = initial_state.copy()
    robot = env.env.robots[0]
    indices = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    qpos = np.asarray(robot.robot_model.init_qpos, dtype=state.dtype)
    nq = int(env.env.sim.model.nq)
    if (indices.ndim != 1 or qpos.shape != indices.shape or len(state) < 1 + nq
            or np.any(indices < 0) or np.any(indices >= nq)):
        raise ValueError("invalid LIBERO-Plus robot initial state")
    state[1 + indices] = qpos
    return state


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """``(x, y, z, w)`` quaternion to an axis-angle vector, as robosuite computes it."""
    quat = np.asarray(quat, dtype=np.float64)
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(1.0 - w * w)
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(w) / denominator


def _libero_observation(obs: dict) -> tuple[list[np.ndarray], np.ndarray]:
    images = [
        np.ascontiguousarray(obs[key][::-1, ::-1], dtype=np.uint8)
        for key in ("agentview_image", "robot0_eye_in_hand_image")
    ]
    state = np.concatenate((
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ))
    return images, state


def _libero_action(raw: np.ndarray, spec: dict) -> np.ndarray:
    action = np.clip(raw, -1.0, 1.0).copy()
    action[-1] = -1.0 if gripper_is_open(raw[-1], spec) else 1.0  # LIBERO: -1 opens
    return action


def run_libero(args: argparse.Namespace, client: PolicyClient) -> dict:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    names = _libero_task_names(args.benchmark_path, args.suite)
    if args.benchmark == "libero-plus" and (args.episode_start, args.episodes) != (0, 1):
        raise ValueError("LIBERO-Plus evaluation uses one official initial state per task")
    task_ids = range(len(names)) if args.task_id is None else [args.task_id]
    bddl_root = Path(get_libero_path("bddl_files"))
    init_root = Path(get_libero_path("init_states"))
    max_steps = args.max_steps or {
        "libero_spatial": 280, "libero_object": 280, "libero_goal": 300,
        "libero_10": 520,
    }.get(args.suite, 520)
    # The environment steps at the rate the policy was trained on (LIBERO: 20 Hz).
    control_freq = int(round(client.metadata["action_spec"]["control_frequency_hz"]))
    successes = attempts = 0
    for task_id in task_ids:
        if not 0 <= task_id < len(names):
            raise ValueError(f"task id {task_id} is outside 0..{len(names) - 1}")
        name = names[task_id]
        bddl = bddl_root / args.suite / f"{name}.bddl"
        if args.benchmark == "libero-plus":
            language, init_path, single_state, robot_qpos = _plus_task_info(
                args.benchmark_path, args.suite, name, bddl_root, init_root)
        else:
            language = _task_language(name)
            init_path = init_root / args.suite / f"{name}.pruned_init"
            single_state = robot_qpos = False
        init_states = _libero_init_states(init_path, single_state=single_state)
        if args.episode_start + args.episodes > len(init_states):
            raise ValueError(f"task {task_id} has only {len(init_states)} fixed initial states")
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=256, camera_widths=256, control_freq=control_freq,
        )
        try:
            env.seed(args.seed)
            for episode_id in range(args.episode_start, args.episode_start + args.episodes):
                env.reset()
                initial_state = init_states[episode_id]
                if robot_qpos:
                    initial_state = _with_robot_qpos(env, initial_state)
                obs = env.set_init_state(initial_state)
                client.reset()
                for _ in range(args.settle_steps):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                success = False
                for _ in range(max_steps):
                    frames, state = _libero_observation(obs)
                    raw = client.next_action(
                        frames, language,
                        state if client.metadata["proprio_mode"] != "none" else None,
                    )
                    obs, _, done, _ = env.step(_libero_action(raw, client.metadata["action_spec"]))
                    if done or env.check_success():
                        success = True
                        break
                successes += int(success)
                attempts += 1
                print(f"{args.suite} task={task_id} episode={episode_id} success={success}", flush=True)
        finally:
            env.close()
    return {"benchmark": args.benchmark, "suite": args.suite,
            "successes": successes, "episodes": attempts, "success_rate": successes / attempts}


def _simplerenv_action(raw: np.ndarray, spec: dict) -> np.ndarray:
    rotation = Rotation.from_euler("xyz", raw[3:6]).as_rotvec()
    gripper = 1.0 if gripper_is_open(raw[-1], spec) else -1.0  # WidowX: +1 opens
    return np.concatenate((raw[:3], rotation, [gripper])).astype(np.float32)


def run_simplerenv(args: argparse.Namespace, client: PolicyClient) -> dict:
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    spec = client.metadata["action_spec"]
    env = simpler_env.make(args.task)
    successes = 0
    try:
        for episode_id in range(args.episode_start, args.episode_start + args.episodes):
            obs, _ = env.reset(seed=args.seed + episode_id)
            client.reset()
            success = False
            task = env.get_language_instruction()
            for _ in range(args.settle_steps):
                obs, _, terminated, _, _ = env.step(
                    np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
                if terminated:
                    success = True
                    break
            for _ in range(args.max_steps):
                if success:
                    break
                image = get_image_from_maniskill2_obs_dict(
                    env, obs,
                )
                raw = client.next_action([np.asarray(image, dtype=np.uint8)], task, None)
                obs, _, terminated, _, _ = env.step(_simplerenv_action(raw, spec))
                if terminated:
                    success = True
                    break
                new_task = env.get_language_instruction()
                if new_task != task:
                    task = new_task
                    client.plan.clear()
            successes += int(success)
            print(f"{args.task} episode={episode_id} success={success}", flush=True)
    finally:
        env.close()
    return {"benchmark": "simplerenv", "environment": args.task,
            "successes": successes, "episodes": args.episodes,
            "success_rate": successes / args.episodes}


def main(argv: list[str] | None = None) -> int:
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--benchmark", choices=("libero", "libero-plus", "simplerenv"))
    selected, _ = selector.parse_known_args(argv)
    parser = argparse.ArgumentParser(
        description="Evaluate a served policy in LIBERO, LIBERO-Plus, or SimplerEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--benchmark", required=True, choices=("libero", "libero-plus", "simplerenv"),
                        help="benchmark to evaluate")
    parser.add_argument("--benchmark-path", required=True, type=Path,
                        help="benchmark repository checkout")
    parser.add_argument("--server", default="ws://127.0.0.1:18080", help="policy server URL")
    parser.add_argument("--episodes", type=int, default=1,
                        help="episodes per task" if selected.benchmark != "libero-plus"
                        else "one official initial state per task; must be 1")
    episode_start_help = "first episode index (initial state index or simulator seed offset)"
    if selected.benchmark == "simplerenv":
        episode_start_help = "first episode seed offset"
    elif selected.benchmark in {"libero", "libero-plus"}:
        episode_start_help = "first official initial state index"
    parser.add_argument("--episode-start", type=int, default=0,
                        help=episode_start_help)
    parser.add_argument("--seed", type=int, default=7, help="simulator random seed")
    parser.add_argument("--max-steps", type=int, help="step limit per episode; benchmark default if omitted")
    parser.add_argument("--settle-steps", type=int, default=10, help="initial no-op steps")
    if selected.benchmark != "simplerenv":
        libero = parser.add_argument_group("LIBERO / LIBERO-Plus options")
        libero.add_argument(
            "--suite", default="libero_spatial",
            choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
            help="task suite in the benchmark checkout",
        )
        libero.add_argument("--task-id", type=int, help="task index; all tasks if omitted")
        libero.add_argument("--libero-config", type=Path,
                            help="directory containing an existing config.yaml; auto-configured if omitted")
    if selected.benchmark not in {"libero", "libero-plus"}:
        simpler = parser.add_argument_group("SimplerEnv options")
        simpler.add_argument(
            "--task", default="widowx_stack_cube",
            choices=("widowx_spoon_on_towel", "widowx_carrot_on_plate",
                     "widowx_stack_cube", "widowx_put_eggplant_in_basket"),
            help="SimplerEnv task name",
        )
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.episode_start < 0 or args.settle_steps < 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error("episodes and max-steps must be positive; episode-start and settle-steps must be non-negative")
    if args.benchmark == "libero-plus" and (args.episode_start, args.episodes) != (0, 1):
        parser.error("LIBERO-Plus uses one official initial state per task: --episodes 1 --episode-start 0")
    if not args.benchmark_path.is_dir():
        parser.error(f"benchmark checkout does not exist: {args.benchmark_path}")
    sys.path.insert(0, str(args.benchmark_path.resolve()))
    with ExitStack() as stack:
        if args.benchmark != "simplerenv":
            config = args.libero_config or args.benchmark_path / ".libero"
            if args.libero_config and not (config / "config.yaml").is_file():
                parser.error(f"LIBERO config.yaml is missing: {config}")
            if not (config / "config.yaml").is_file():
                config = Path(stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="actioncodec2-libero-")
                ))
                root = args.benchmark_path.resolve() / "libero/libero"
                paths = {"benchmark_root": root, "bddl_files": root / "bddl_files",
                         "init_states": root / "init_files", "assets": root / "assets",
                         "datasets": root.parent / "datasets"}
                (config / "config.yaml").write_text(
                    "".join(f"{key}: {value}\n" for key, value in paths.items()),
                    encoding="utf-8",
                )
            os.environ["LIBERO_CONFIG_PATH"] = str(config.resolve())
        if args.benchmark == "simplerenv" and args.max_steps is None:
            args.max_steps = 240
        client = PolicyClient(args.server)
        try:
            require_contract(client.metadata, benchmark=args.benchmark)
            result = run_simplerenv(args, client) if args.benchmark == "simplerenv" else run_libero(args, client)
            print(json.dumps(result, sort_keys=True), flush=True)
        finally:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
