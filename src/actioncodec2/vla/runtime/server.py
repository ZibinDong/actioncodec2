"""Serve a policy checkpoint to simulator clients over a WebSocket.

Launch with ``python scripts/serve.py --model <directory or Hub repository ID>``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
import traceback
import queue
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
from websockets.sync.server import serve

MAX_ARRAY_BYTES = 64 * 1024 * 1024


def pack_array(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.nbytes > MAX_ARRAY_BYTES:
        raise ValueError("action array exceeds the WebSocket size limit")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def unpack_array(payload: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    if dtype.kind not in "biuf":
        raise ValueError("observation array must have a numeric dtype")
    shape = tuple(int(size) for size in payload["shape"])
    if any(size < 0 for size in shape):
        raise ValueError("negative array dimension")
    expected = dtype.itemsize
    for size in shape:
        expected *= size
        if expected > MAX_ARRAY_BYTES:
            raise ValueError("observation array exceeds the WebSocket size limit")
    if expected > MAX_ARRAY_BYTES:
        raise ValueError("observation array exceeds the WebSocket size limit")
    raw = base64.b64decode(payload["data"], validate=True)
    if len(raw) != expected:
        raise ValueError("observation array has an invalid byte count")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def action_spec(data) -> dict[str, Any]:
    """Physical meaning of the action chunks a checkpoint emits (from ``DataConfig``)."""
    low, high = data.gripper_range
    spec: dict[str, Any] = {
        "space": data.action_space,
        "frame": data.action_frame,
        "control_frequency_hz": data.control_frequency_hz,
        "gripper_representation": "scalar",
        "gripper_open": high if data.gripper_open_is_high else low,
        "gripper_closed": low if data.gripper_open_is_high else high,
        "gripper_binary_threshold": data.gripper_threshold,
    }
    if data.action_space.endswith("eef"):
        spec.update(
            dimension=7,
            rotation_representation=data.action_rotation_representation,
            # One stored unit in meters and radians; absent when the dataset does
            # not say (a client must not assume physical units then).
            translation_meters_per_unit=data.translation_meters_per_unit,
            rotation_radians_per_unit=data.rotation_radians_per_unit,
        )
    else:
        spec.update(dimension=8, joint_unit="radian")
    return spec


def execution_horizon(stride: int | None, *, stateful: bool, horizon: int) -> int:
    """Return the configured stride, otherwise one step for stateful tokenizers or the full horizon."""
    return int(stride or (1 if stateful else horizon))


class MicroBatcher:
    """Run concurrent single-observation requests as one batched policy call."""

    def __init__(self, policy, *, device: str, max_batch_size: int, wait_seconds: float) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.policy = policy
        self.max_batch_size = int(max_batch_size)
        self.wait_seconds = float(wait_seconds)
        self.requests: queue.Queue = queue.Queue()
        self.device = device
        threading.Thread(target=self._loop, name="policy", daemon=True).start()

    def submit(self, observation: dict[str, Any], temperature: float) -> Future:
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        future: Future = Future()
        self.requests.put((observation, temperature, future))
        return future

    def _loop(self) -> None:
        if str(self.device).startswith("cuda"):
            import torch

            torch.cuda.set_device(torch.device(self.device))
        while True:
            pending = [self.requests.get()]
            deadline = time.monotonic() + self.wait_seconds
            while len(pending) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                try:
                    pending.append(
                        self.requests.get(timeout=remaining) if remaining > 0
                        else self.requests.get_nowait()
                    )
                except queue.Empty:
                    break
            try:
                for temperature in dict.fromkeys(temperature for _, temperature, _ in pending):
                    group = [item for item in pending if item[1] == temperature]
                    self._run(group, temperature)
            except Exception as error:
                # Whatever went wrong, never leave a client waiting on a dead thread.
                for _, _, future in pending:
                    if not future.done():
                        future.set_exception(error)

    def _run(self, group: list, temperature: float) -> None:
        try:
            results = self.policy.predict_batch(
                [observation for observation, _, _ in group], temperature=temperature
            )
            if len(results) != len(group):
                raise RuntimeError("the policy returned the wrong number of results")
        except Exception as error:
            if len(group) == 1:
                group[0][2].set_exception(error)
                return
            # Attribute the failure to the request that caused it.
            for item in group:
                self._run([item], temperature)
            return
        for (_, _, future), result in zip(group, results, strict=True):
            future.set_result(result)


class PolicyRuntime:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str,
        max_batch_size: int = 32,
        batch_wait_ms: float = 3.0,
        dtype: str = "float32",
        revision: str | None = None,
        trust_remote_code: bool = False,
        request_timeout: float = 600.0,
    ) -> None:
        from actioncodec2.vla.runtime import DTYPES, load_policy
        from actioncodec2.vla.runtime.run import load_runtime_config, resolve_checkpoint

        self.request_timeout = float(request_timeout)
        self.checkpoint = resolve_checkpoint(
            checkpoint, revision=revision, trust_remote_code=trust_remote_code
        ).resolve()
        config = load_runtime_config(self.checkpoint)
        self.policy, _ = load_policy(self.checkpoint, device=device, dtype=DTYPES[dtype])
        stride = config.data.execution_stride
        spec = action_spec(config.data)
        self.metadata = {
            "checkpoint": self.checkpoint.name,
            "policy_config_sha256": hashlib.sha256(
                (self.checkpoint / "policy_config.json").read_bytes()
            ).hexdigest(),
            "device": device,
            "action_horizon": config.data.horizon,
            "execution_horizon": execution_horizon(
                stride,
                stateful=self.policy.requires_previous_action,
                horizon=config.data.horizon,
            ),
            "trained_execution_stride": stride,
            # What a client must send: the tokenizer's boundary, or the model's own
            # action-history input when the policy was trained with one.
            "requires_previous_action": self.policy.needs_previous_action_input,
            "proprio_mode": config.model.proprio_mode,
            "state_dim": (self.policy.processor.proprio_transform.dimension
                          if config.model.proprio_mode == "observed" else 0),
            "camera_keys": config.data.video_keys,
            "action_spec": spec,
            "image_size": list(self.policy.processor.image_size),
        }
        run_config = self.checkpoint / "run_config.json"
        if run_config.is_file():
            self.metadata["run_config_sha256"] = hashlib.sha256(
                run_config.read_bytes()
            ).hexdigest()
        self.batcher = MicroBatcher(
            self.policy, device=device, max_batch_size=max_batch_size,
            wait_seconds=batch_wait_ms / 1000.0,
        )

    @staticmethod
    def observation(payload: dict[str, Any]) -> dict[str, Any]:
        frames = [unpack_array(frame) for frame in payload["frames"]]
        if not frames or any(
            frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3
            for frame in frames
        ):
            raise ValueError("frames must contain uint8 HWC RGB images")
        state = payload.get("state")
        previous = payload.get("previous_action")
        return {
            "frames": frames,
            "task": str(payload["task"]),
            "state": None if state is None else np.asarray(state, dtype=np.float32),
            "previous_action": None if previous is None else np.asarray(previous, dtype=np.float32),
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.observation(payload)
        started = time.monotonic()
        future = self.batcher.submit(row, float(payload.get("temperature", 0)))
        actions, tokens = future.result(timeout=self.request_timeout)
        return {
            "actions": pack_array(np.asarray(actions, dtype=np.float32)),
            "tokens": [int(token) for token in tokens],
            "inference_seconds": time.monotonic() - started,
        }

    def predict_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows = payload["observations"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("predict_batch requires observations")
        observations = [self.observation(row) for row in rows]
        temperature = float(payload.get("temperature", 0))
        started = time.monotonic()
        futures = [self.batcher.submit(observation, temperature) for observation in observations]
        predictions = [future.result(timeout=self.request_timeout) for future in futures]
        return {
            "results": [
                {"actions": pack_array(np.asarray(actions, dtype=np.float32)),
                 "tokens": [int(token) for token in tokens]}
                for actions, tokens in predictions
            ],
            "batch_size": len(predictions),
            "inference_seconds": time.monotonic() - started,
        }


def make_handler(runtime: PolicyRuntime):
    def handler(connection) -> None:
        for message in connection:
            request_id = None
            try:
                if not isinstance(message, str) or len(message.encode()) > MAX_ARRAY_BYTES:
                    raise ValueError("invalid or oversized WebSocket request")
                request = json.loads(message)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                request_id = request.get("id")
                method = request.get("method")
                params = request.get("params", {})
                if not isinstance(params, dict):
                    raise ValueError("params must be an object")
                if method == "health":
                    result = {"ready": True, **runtime.metadata}
                elif method == "metadata":
                    result = runtime.metadata
                elif method == "predict":
                    result = runtime.predict(params)
                elif method == "predict_batch":
                    result = runtime.predict_batch(params)
                else:
                    raise ValueError(f"unknown RPC method: {method!r}")
                response = {"id": request_id, "result": result}
            except Exception:
                traceback.print_exc()
                response = {"id": request_id, "error": "request failed; see server logs"}
            connection.send(json.dumps(response, separators=(",", ":")))
    return handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="Hugging Face model repository ID or local model directory")
    model.add_argument("--checkpoint", dest="model", help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-batch-size", type=int, default=32,
        help="most concurrent requests decoded together",
    )
    parser.add_argument(
        "--batch-wait-ms", type=float, default=3.0,
        help="how long the first request of a batch waits for others to join",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32",
        help="weight dtype (default: float32 with bfloat16 autocast)",
    )
    parser.add_argument(
        "--revision", help="optional Hub model version: branch, tag, or commit (default: main)"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="allow bundled tokenizer code to run (required for Hub models)",
    )
    args = parser.parse_args(argv)
    runtime = PolicyRuntime(
        args.model, device=args.device, max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms, dtype=args.dtype, revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    with serve(make_handler(runtime), args.host, args.port,
               max_size=MAX_ARRAY_BYTES, ping_interval=None) as server:
        print(f"policy ready at ws://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0
