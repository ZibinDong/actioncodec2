"""Export a trained VLA checkpoint into a portable Hugging Face model directory."""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from actioncodec2.publishing import audit_public_files, prepare_public_assets
from actioncodec2.tokenizers.base import QuantileNormalizer
from actioncodec2.tokenizers.factory import DEFAULT_ACTIONCODEC, DEFAULT_FAST, action_bounds
from actioncodec2.vla.g05_hf.constants import G05_MODEL_TYPE, LEGACY_MODEL_TYPES
from actioncodec2.vla.training.config import RunConfig


ASSET_FIELDS = {
    "tokenizer": (
        "artifact", "processor", "action_stats", "normalizer_path",
        "statistics_path", "calibration_path",
    ),
}
#: Owner shown in the model card when ``--repo-id`` is not given.
PLACEHOLDER_OWNER = "your-org"
SOURCE_ARCHIVE_URL = (
    "https://hf-mirror.com/ZibinDong/ActionCodec2/resolve/v0.1.0/actioncodec2-0.1.0.tar.gz"
)
CHECKPOINT_FILES = (
    "config.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
    "generation_config.json", "model.safetensors.index.json",
)
POLICY_FIELDS = (
    "action_vocab_size", "action_token_begin", "total_vocab_size",
    "g05_proprio_stats",
)
MODEL_INFERENCE_FIELDS = (
    "embodiment", "dtype", "attn_implementation", "image_size", "action_dim",
    "proprio_mode",
)
ACTION_INFERENCE_FIELDS = (
    "horizon", "action_space", "action_frame", "action_rotation_representation",
    "control_frequency_hz", "gripper_range", "gripper_threshold", "gripper_open_is_high",
    "video_keys", "translation_meters_per_unit", "rotation_radians_per_unit",
)


def inference_policy_config(config: dict, saved_policy_config: dict) -> dict:
    """Keep only fields needed to rebuild a policy and describe its actions."""
    model = {key: config["model"][key] for key in MODEL_INFERENCE_FIELDS}
    if model["proprio_mode"] != "none":
        model["proprio_source_indices"] = config["model"]["proprio_source_indices"]
        model["proprio_quantile_indices"] = config["model"]["proprio_quantile_indices"]
        if config["model"].get("proprio_internal_indices") is not None:
            model["proprio_internal_indices"] = config["model"]["proprio_internal_indices"]
    if config["model"].get("force_previous_action_input") is not None:
        model["force_previous_action_input"] = config["model"]["force_previous_action_input"]
    action = {
        key: config["data"][key] for key in ACTION_INFERENCE_FIELDS
        if config["data"].get(key) is not None
    }
    # The stride the policy was trained with: the whole chunk for stateless
    # tokenizers, the boundary stride for stateful ones.
    stride = saved_policy_config.get("execution_stride") or config["data"].get("execution_stride")
    if stride is not None:
        action["execution_stride"] = int(stride)
    tokenizer = dict(config["tokenizer"])
    tokenizer.pop("device", None)
    redundant = {
        "time_horizon": action["horizon"],
        "action_dim": model["action_dim"],
        "fps": action.get("control_frequency_hz") or 20.0,
        "gripper_open_is_high": action["gripper_open_is_high"],
    }
    if action["action_space"].endswith("eef"):
        redundant["rotation_representation"] = action["action_rotation_representation"]
    if str(tokenizer.get("name", "")).startswith("actioncodec2"):
        redundant["gripper_threshold"] = action["gripper_threshold"]
        redundant["action_space"] = action["action_space"]
    for key, value in redundant.items():
        if tokenizer.get(key) == value:
            tokenizer.pop(key)
    policy_config = {
        "format_version": 1,
        "model": model,
        "tokenizer": tokenizer,
        "action": action,
        **{
            key: saved_policy_config[key] for key in POLICY_FIELDS
            if key in saved_policy_config
        },
    }
    if config["model"]["proprio_mode"] == "observed":
        stats = policy_config["g05_proprio_stats"]
        policy_config["g05_proprio_stats"] = {
            key: stats[key] for key in ("train_mean", "train_std", "q01", "q99")
        } | {"quantile_indices": stats.get(
            "quantile_indices", config["model"]["proprio_quantile_indices"]
        )}
    else:
        policy_config.pop("g05_proprio_stats", None)
    return policy_config


def _external_tokenizer(config: dict) -> bool:
    """Inline normalization assets while preserving the tokenizer reference."""
    settings = config["tokenizer"]
    name = str(settings.get("name", "")).lower().replace("_", "-")
    stats_path = settings.get("action_stats")
    if stats_path:
        stats = json.loads(_asset(str(stats_path), "tokenizer.action_stats").read_text())
        threshold = stats.get("gripper_threshold")
        if threshold is not None and float(threshold) != float(config["data"]["gripper_threshold"]):
            raise ValueError("tokenizer.action_stats gripper threshold differs from data")

    if name in {"actioncodec2", "actioncodec2-sparse", "ac2sp", "actioncodec2sparse"}:
        if stats_path or "action_clip" in settings:
            low, high = action_bounds(settings)
            settings["action_clip"] = {"low": low.tolist(), "high": high.tolist()}
            settings.pop("action_stats", None)
            settings.pop("action_stats_section", None)
        return False
    if name == "x-tokenizer":
        if settings.get("normalizer_path"):
            payload = json.loads(_asset(settings["normalizer_path"], "tokenizer.normalizer_path").read_text())
            settings["normalizer"] = QuantileNormalizer.from_json(payload).to_json()
            settings.pop("normalizer_path")
        elif "normalizer" in settings:
            settings["normalizer"] = QuantileNormalizer.from_json(settings["normalizer"]).to_json()
        return False
    if name not in {"fast", "actioncodec"}:
        return False

    if name == "fast":
        if settings.get("normalizer_path"):
            payload = json.loads(
                _asset(str(settings["normalizer_path"]), "tokenizer.normalizer_path").read_text()
            )
            settings["normalizer"] = QuantileNormalizer.from_json(payload).to_json()
        elif "normalizer" not in settings and stats_path:
            settings["normalizer"] = QuantileNormalizer(*action_bounds(settings)).to_json()
        if "normalizer" not in settings:
            raise ValueError("FAST requires normalizer, normalizer_path or action_stats")
        settings["normalizer"] = QuantileNormalizer.from_json(settings["normalizer"]).to_json()
        reference = settings.setdefault("processor", settings.get("source", DEFAULT_FAST))
        settings.pop("source", None)
        settings.pop("artifact", None)
    else:
        if "action_clip" not in settings and not stats_path:
            raise ValueError("ActionCodec requires action_clip or action_stats")
        low, high = action_bounds(settings)
        settings["action_clip"] = {"low": low.tolist(), "high": high.tolist()}
        reference = settings.setdefault("artifact", DEFAULT_ACTIONCODEC)
        settings.pop("processor", None)
    for key in (
        "action_stats", "action_stats_section", "normalizer_path",
        "statistics_path", "calibration_path",
    ):
        settings.pop(key, None)
    return not Path(reference).expanduser().exists() and not str(reference).startswith(
        ("/", "./", "../", "~")
    )


def finalize_bundle(bundle: Path, repo_id: str | None = None) -> None:
    """Write the published model configuration, card, and license notices."""
    policy_path = bundle / "policy_config.json"
    policy_config = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy_config.get("format_version") != 1:
        raise ValueError("bundle requires a versioned inference policy_config.json")
    model_path = bundle / "config.json"
    model_config = json.loads(model_path.read_text(encoding="utf-8"))
    if model_config.get("model_type") in LEGACY_MODEL_TYPES:
        model_config["model_type"] = G05_MODEL_TYPE
    elif model_config.get("model_type") != G05_MODEL_TYPE:
        raise ValueError(f"unrecognized G0.5 model_type in {model_path}")
    model_config["architectures"] = ["G05Model"]
    model_path.write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    shutil.copy2(project_root / "LICENSE-G0.5", bundle / "LICENSE")
    shutil.copy2(project_root / "LICENSE_QWEN3_5.txt", bundle / "LICENSE_QWEN3_5.txt")
    (bundle / "NOTICE").write_text(
        "G0.5 — Notice\n\n"
        "G0.5 is licensed under the G0.5 Community License Agreement "
        "(Non-Commercial + Limited Patent License), not sold, Copyright © 2026 "
        "Galaxea. All rights reserved by Galaxea. ‘Galaxea’ and related marks "
        "are trademarks of Galaxea or its affiliates.\n"
        "This repository may include third-party components under their own "
        "licenses. You are responsible for complying with applicable "
        "third-party license terms.\n\n"
        "This release modifies G0.5: the model weight files contain fine-tuned "
        "weights, and config.json contains the policy configuration. The "
        "ActionCodec2 project prepared these modifications; Galaxea does not "
        "endorse this release. The included Qwen3.5 text tokenizer is covered "
        "by LICENSE_QWEN3_5.txt.\n",
        encoding="utf-8",
    )
    for old_name in ("LICENSE-G0.5", "_SUCCESS"):
        (bundle / old_name).unlink(missing_ok=True)
    repo_id = repo_id or f"{PLACEHOLDER_OWNER}/{policy_repo_name(policy_config)}"
    (bundle / "README.md").write_text(model_card(policy_config, repo_id), encoding="utf-8")


def policy_label(policy_config: dict) -> str:
    """Tokenizer part of a released policy's name, e.g. ``ActionCodec2-2nd``."""
    tokenizer = policy_config["tokenizer"]
    name = tokenizer["name"]
    bundled = (str(tokenizer.get("artifact")).startswith(("./", "../"))
               and not tokenizer.get("variant"))
    if name == "x-tokenizer":
        return "X-Tokenizer-finetuned" if bundled else "X-Tokenizer"
    if name != "actioncodec2":
        return {"actioncodec": "ActionCodec", "fast": "FAST", "actioncodec2-sparse": "ActionCodec2Sparse"}.get(name, name)
    stateful = policy_config.get("requires_previous_action") or tokenizer.get(
        "previous_action_resampling"
    )
    return f"ActionCodec2{'-corpusfit' if bundled else ''}-{'2nd' if stateful else '1st'}"


def benchmark_name(policy_config: dict) -> str | None:
    """The public benchmark a policy's embodiment belongs to, or ``None`` for custom data."""
    embodiment = policy_config["model"]["embodiment"].lower()
    if embodiment == "libero":
        return "LIBERO"
    if "droid" in embodiment:
        return "DROID"
    if embodiment == "bridge":
        return "SimplerEnv"
    return None


def policy_repo_name(policy_config: dict) -> str:
    """Hub repository name: ``G05_<tokenizer>_<benchmark or embodiment>``."""
    benchmark = benchmark_name(policy_config) or re.sub(
        r"[^A-Za-z0-9]+", "-", policy_config["model"]["embodiment"]
    ).strip("-")
    return f"G05_{policy_label(policy_config)}_{benchmark}"


def model_card(policy_config: dict, repo_id: str) -> str:
    """Describe the policy's provenance, bundled tokenizer, and observation contract."""
    label = policy_label(policy_config)
    benchmark = benchmark_name(policy_config)
    dataset = "Bridge" if benchmark == "SimplerEnv" else (
        benchmark or policy_config["model"]["embodiment"]
    )
    stateful = bool(policy_config.get("requires_previous_action") or
                    policy_config["model"].get("force_previous_action_input"))
    state = policy_config.get("g05_proprio_stats", {}).get("train_mean")
    action = policy_config["action"]
    model = policy_config["model"]
    variant = policy_config["tokenizer"].get("variant")
    source_archive_url = SOURCE_ARCHIVE_URL
    call = "policy.predict(frames, task" + (", state=state" if state else "") + (
        ", previous_action=previous_action" if stateful else "") + ")"
    example_inputs = (
        "# Synthetic inputs for checking the interface; replace with current observations.\n"
        "height, width = policy_config[\"model\"][\"image_size\"]\n"
        "frames = [np.zeros((height, width, 3), dtype=np.uint8)\n"
        "          for _ in policy_config[\"action\"][\"video_keys\"]]\n"
        "task = \"pick up the object\"\n"
    )
    if state:
        example_inputs += f"state = np.zeros({len(state)}, dtype=np.float32)\n"
    if stateful:
        example_inputs += (
            "# This synthetic example assumes zero preceding motion.\n"
            f"previous_action = np.zeros({model['action_dim']}, dtype=np.float32)\n"
        )
    bundled_note = (
        "The checkpoint includes its action tokenizer, fitted assets, and text tokenizer. "
        "After downloading the complete repository, inference works offline without "
        "the G0.5 base checkpoint or another tokenizer repository.\n\n"
        if str(policy_config["tokenizer"].get("artifact", "")).startswith("./") else ""
    )
    tokenizer_note = (
        f"This policy uses **`{variant}`**, the **5 mm Paper-edition** tokenizer used in "
        "the ActionCodec2 paper, with a **4,096-token budget per profile**. The separate "
        "Extended edition was retrained after additional cleaning and filtering of the "
        "pretraining data and offers more precision and vocabulary options. Keep this "
        "policy's bundled vocabulary: token IDs are tied to its trained weights.\n\n"
        if variant in {"paper-1st", "paper-2nd", "paper-o1-5mm-4k-2bin-1p2ms", "paper-o2-5mm-4k-2bin-1p2ms"} else ""
    )
    boundary_note = (
        f"Pass the last executed raw action, a vector of length {model['action_dim']}, "
        "as `previous_action` on every prediction. Use the same units and gripper "
        "convention as the returned actions. A zero motion boundary is valid only "
        "when the preceding motion is known to be zero. "
        + ("The adapter converts this action to the tokenizer's 15 Hz boundary. "
           if policy_config["tokenizer"]["name"] == "actioncodec2" else "") + "\n\n"
        if stateful else ""
    )
    state_note = ""
    if state:
        state_note = f"Pass all {len(state)} raw values as `state`; the processor selects and normalizes them. "
        if benchmark == "LIBERO":
            state_note += (
                "The order is end-effector position (3), orientation as a rotation vector (3), "
                "and both gripper finger positions (2). "
            )
        elif benchmark == "DROID":
            state_note += "The order is seven joint positions followed by the gripper state. "
        state_note += "\n\n"
    camera_names = ", ".join(f"`{key}`" for key in (action.get("video_keys") or ()))
    rotation_row = (
        f"| Rotation representation | `{action['action_rotation_representation']}` |\n"
        if action["action_space"].endswith("eef") else ""
    )
    evaluation_note = (
        " It targets **WidowX evaluation in SimplerEnv**; Bridge is the training dataset."
        if benchmark == "SimplerEnv" else ""
    )
    tokenizer_license = (
        "Bundled ActionCodec2 assets use [Apache-2.0](assets/tokenizer_artifact/LICENSE). "
        if policy_config["tokenizer"]["name"] == "actioncodec2" and bundled_note else ""
    )
    return (
        "---\n"
        "license: other\n"
        "license_name: g05-community\n"
        "license_link: LICENSE\n"
        "base_model: OpenGalaxea/G05\n"
        "base_model_relation: finetune\n"
        "tags:\n"
        "  - robotics\n"
        "  - vision-language-action\n"
        f"  - {policy_config['tokenizer']['name']}\n"
        "---\n\n"
        f"# {repo_id.split('/')[-1]}\n\n"
        f"[G0.5](https://hf-mirror.com/OpenGalaxea/G05) fine-tuned on **{dataset}** with "
        f"the **{label}** action tokenizer.{evaluation_note}\n\n"
        f"{tokenizer_note}{bundled_note}"
        "## Installation\n\n"
        "Use **Python 3.12 or later**, a C++17 compiler, and a CUDA-capable GPU for the "
        "example below. Inference has been checked with Python 3.12, PyTorch 2.7.1, "
        "and Transformers 5.14.1.\n\n"
        "```bash\n"
        "export HF_ENDPOINT=https://hf-mirror.com\n"
        f"pip install \"actioncodec2[cuda] @ {source_archive_url}\"\n"
        "```\n\n"
        f"The [source archive]({source_archive_url}) also contains training recipes, "
        "evaluation clients, and their documentation. The `cuda` extra adds Liger "
        "and Flash Linear Attention.\n\n"
        "## Inference\n\n"
        "This example checks loading and the prediction interface with synthetic inputs. "
        "For evaluation or robot control, provide current camera images, a task instruction, "
        "and the required observed state and previous action.\n\n"
        "```python\n"
        "import numpy as np\n"
        "from actioncodec2.vla.runtime import load_policy\n\n"
        "policy, policy_config = load_policy(\n"
        f"    \"{repo_id}\", revision=\"v0.1.0\",\n"
        "    device=\"cuda\", trust_remote_code=True,\n"
        ")\n"
        f"{example_inputs}"
        f"actions, tokens = {call}\n"
        "print(actions.shape)\n"
        "```\n\n"
        "A downloaded directory can be passed to `load_policy` directly. "
        "Use `revision=\"<commit>\"` to pin an exact Hub snapshot.\n\n"
        "## Input and action conventions\n\n"
        "Supply a list of RGB `uint8` H×W×3 images, one per camera in the order below. "
        "The processor resizes them to the configured image size.\n\n"
        "| Setting | Value |\n"
        "| --- | --- |\n"
        f"| Cameras, in order | {camera_names} |\n"
        f"| Model image size | {model['image_size'][0]} × {model['image_size'][1]} |\n"
        f"| Returned action shape | `({action['horizon']}, {model['action_dim']})` |\n"
        f"| Control frequency | {action.get('control_frequency_hz', 20):g} Hz |\n"
        f"| Action space | `{action['action_space']}` |\n"
        f"{rotation_row}"
        f"| Execute before replanning | {action.get('execution_stride') or action['horizon']} actions |\n"
        f"| Observed state | {str(len(state)) + ' raw values' if state else 'Not required'} |\n\n"
        f"{state_note}{boundary_note}"
        "Use the checkpoint's [policy_config.json](policy_config.json) for action scaling, "
        "gripper settings, and the complete inference contract. The evaluation clients "
        "in the source archive perform the benchmark-specific action conversion.\n\n"
        "## License\n\n"
        "Modified G0.5 weights under the [G0.5 Community License](LICENSE) "
        "(non-commercial; no commercial deployment rights). See [NOTICE](NOTICE). "
        f"{tokenizer_license}"
        "The Qwen3.5 text tokenizer is covered by [LICENSE_QWEN3_5.txt](LICENSE_QWEN3_5.txt).\n"
    )


def _asset(value: str, field: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(f"{field} is missing: {value}")


def export_checkpoint(
    checkpoint: str | Path, output: str | Path, repo_id: str | None = None,
    *, tokenizer_catalog: str | Path | None = None,
) -> Path:
    source = Path(checkpoint).resolve()
    target = Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"checkpoint is missing: {source}")
    if target.exists():
        raise FileExistsError(f"bundle already exists: {target}")
    if target.is_relative_to(source):
        raise ValueError("bundle output cannot be inside its checkpoint")
    for name in ("policy_config.json", "config.json", "tokenizer.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"checkpoint file is missing: {source / name}")
    weights = [
        path for path in source.iterdir()
        if path.is_file() and (
            path.name == "model.safetensors"
            or (path.name.startswith("model-") and path.suffix == ".safetensors")
        )
    ]
    if not weights:
        raise FileNotFoundError(f"checkpoint has no safetensors weights: {source}")

    saved_policy_config = json.loads((source / "policy_config.json").read_text(encoding="utf-8"))
    if saved_policy_config.get("format_version") == 1:
        from actioncodec2.vla.runtime.run import load_runtime_config

        config = asdict(load_runtime_config(source))
    else:
        config = RunConfig.load(source / "run_config.json").to_json()
    source_model_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if source_model_config.get("model_type") not in (*LEGACY_MODEL_TYPES, G05_MODEL_TYPE):
        raise ValueError("checkpoint config.json is not a supported G0.5 model")
    for key in ("action_vocab_size", "action_token_begin", "total_vocab_size"):
        if key not in saved_policy_config:
            raise ValueError(f"checkpoint policy_config.json is missing {key}")
    stats = saved_policy_config.get("g05_proprio_stats")
    if config["model"]["proprio_mode"] == "observed":
        keys = ("train_mean", "train_std", "q01", "q99")
        if not isinstance(stats, dict) or any(key not in stats for key in keys):
            raise ValueError("observed proprio requires g05_proprio_stats normalization vectors")
    external_tokenizer = _external_tokenizer(config)
    codec = None
    if config["tokenizer"].get("name") == "actioncodec2":
        from actioncodec2.tokenizers.hub_catalog import ActionCodec2, asset_inventory
        from actioncodec2.core.catalog import resolve_variant_name

        settings = config["tokenizer"]
        reference = settings["artifact"]
        if tokenizer_catalog is not None:
            import numpy as np
            import yaml

            catalog = _asset(str(tokenizer_catalog), "tokenizer_catalog")
            records = json.loads((catalog / "variants.json").read_text())["variants"]
            variant = settings.get("variant")
            if variant is None:
                matches = [name for name, record in records.items()
                           if reference == record.get("legacy_repo_id") or
                           Path(reference).name == str(record.get("legacy_repo_id", "")).split("/")[-1]]
                if len(matches) != 1:
                    raise ValueError("tokenizer_catalog requires an unambiguous tokenizer.variant")
                variant = matches[0]
            codec = ActionCodec2.from_pretrained(
                catalog, variant=variant, local_files_only=True,
            )
            original = Path(reference).expanduser()
            if not original.is_dir():
                original_codec = ActionCodec2.from_pretrained(
                    reference, variant=settings.get("variant"), revision=settings.get("revision"),
                )
                original = original_codec._source_directory
            if (original / "variants.json").is_file():
                original_records = json.loads((original / "variants.json").read_text())["variants"]
                original = original / "variants" / resolve_variant_name(variant, original_records)
            replacement = catalog / "variants" / resolve_variant_name(variant, records)
            for name in asset_inventory(replacement):
                if not name.startswith("profiles/") or Path(name).name not in {"config.yaml", "primitives.npz", "bpe.json"}:
                    continue
                old, new = original / name, replacement / name
                if name.endswith(".npz"):
                    with np.load(old, allow_pickle=False) as left, np.load(new, allow_pickle=False) as right:
                        identical = sorted(left.files) == sorted(right.files) and all(
                            left[key].dtype == right[key].dtype and
                            np.array_equal(left[key], right[key]) for key in left.files
                        )
                elif name.endswith(".json"):
                    identical = json.loads(old.read_text()) == json.loads(new.read_text())
                else:
                    identical = yaml.safe_load(old.read_text()) == yaml.safe_load(new.read_text())
                if not identical:
                    raise ValueError(f"tokenizer catalog changes trained assets: {name}")
        else:
            codec = ActionCodec2.from_pretrained(
                reference, variant=settings.get("variant"), revision=settings.get("revision"),
                variant_fingerprint=settings.get("variant_fingerprint"),
            )
        if codec.variant is not None:
            settings.update(variant=codec.variant, variant_fingerprint=codec.variant_fingerprint)
        settings.pop("revision", None)
    elif config["tokenizer"].get("name") in {"actioncodec2-sparse", "actioncodec2_sparse", "ac2sp", "actioncodec2sparse"}:
        from actioncodec2.tokenizers.factory import build_tokenizer
        from actioncodec2.vla.runtime.run import InferenceConfig, tokenizer_settings
        from actioncodec2.vla.training.config import DataConfig, ModelConfig

        if tokenizer_catalog is not None:
            raise ValueError("tokenizer_catalog applies only to dense ActionCodec2 policies")
        inference = InferenceConfig(
            ModelConfig(**config["model"]), config["tokenizer"], DataConfig(**config["data"])
        )
        adapter = build_tokenizer(tokenizer_settings(inference))
        codec = adapter.codec
        config["tokenizer"] = adapter.to_json()
        config["tokenizer"].pop("revision", None)
    elif tokenizer_catalog is not None:
        raise ValueError("tokenizer_catalog applies only to ActionCodec2 policies")
    referenced = set()
    for key in ("artifact", "processor"):
        value = config["tokenizer"].get(key)
        if value and not Path(value).expanduser().exists() and not str(value).startswith(
            ("/", "./", "../", "~")
        ):
            referenced.add(key)
    assets = {}
    for group, fields in ASSET_FIELDS.items():
        if external_tokenizer and group == "tokenizer":
            continue
        for key in fields:
            if group == "tokenizer" and key == "artifact" and codec is not None:
                continue
            if group == "tokenizer" and key in referenced:
                continue
            value = config[group].get(key)
            if value:
                assets[(group, key)] = _asset(str(value), f"{group}.{key}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.export-", dir=target.parent))
    owned_target = False
    try:
        for name in CHECKPOINT_FILES:
            path = source / name
            if path.is_file():
                shutil.copy2(path, temporary / name)
        for path in weights:
            shutil.copy2(path, temporary / path.name)
        if codec is not None:
            relative = Path("assets/tokenizer_artifact")
            codec.save_pretrained(temporary / relative)
            config["tokenizer"]["artifact"] = "./" + relative.as_posix()
        for (group, key), path in assets.items():
            if path.is_relative_to(source):
                relative = path.relative_to(source)
            else:
                relative = Path("assets") / f"{group}_{key}"
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, destination, ignore=shutil.ignore_patterns(
                    ".cache", ".git", "__pycache__", "*.pyc", "*.so", "SOURCE.json",
                    ".pi", ".claude", ".codex", ".agents", ".cursor", ".opencode", ".aider*",
                    "AGENTS.md", "CLAUDE.md", "CODEX.md", "GEMINI.md", "SKILL.md",
                    ".idea", ".vscode", ".DS_Store", "*.swp", "*.swo", ".env", ".env.*",
                    "*.log", "*.jsonl", "wandb", "swanlog",
                ))
            else:
                shutil.copy2(path, destination)
            config[group][key] = "./" + relative.as_posix()
        # Checkpoints written before X-Tokenizer became a package dependency.
        config["tokenizer"].pop("source_path", None)
        policy_config = inference_policy_config(config, saved_policy_config)
        # The G0.5 head only takes a boundary action for stateful tokenizers.
        policy_config["requires_previous_action"] = bool(
            source_model_config.get("continuous_action_dim", 0)
        )
        (temporary / "policy_config.json").write_text(
            json.dumps(policy_config, indent=2) + "\n", encoding="utf-8"
        )
        prepare_public_assets(temporary)
        finalize_bundle(temporary, repo_id)
        try:
            os.replace(temporary, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            # Some network filesystems reject directory rename even when the
            # temporary directory is created beside its final destination.
            target.mkdir()
            owned_target = True
            shutil.copytree(temporary, target, dirs_exist_ok=True)
            shutil.rmtree(temporary)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if owned_target and target.exists():
            shutil.rmtree(target)
        raise
    return target


def verify_bundle(bundle: Path) -> None:
    """Rebuild the action tokenizer from the exported files alone."""
    import numpy as np
    from actioncodec2.vla.runtime.run import build_tokenizer_for, load_runtime_config

    config = load_runtime_config(bundle)
    tokenizer = build_tokenizer_for(config)
    actions = np.zeros((2, config.data.horizon, config.model.action_dim), np.float32)
    boundary = np.zeros((2, config.model.action_dim), np.float32) if tokenizer.requires_previous_action else None
    tokens = tokenizer.encode_batch(actions, boundary)
    decoded = tokenizer.decode_batch(tokens, boundary)
    if (decoded.shape != actions.shape or not np.isfinite(decoded).all() or
            not all(tokenizer.grammar.is_complete(row) for row in tokens)):
        raise ValueError("exported tokenizer failed its encode/decode contract")
    print(
        f"verified: {tokenizer.name}, vocab {tokenizer.vocab_size}, "
        f"{tokenizer.max_tokens_per_chunk} tokens per chunk at most"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint", type=Path)
    inputs.add_argument("--release-root", type=Path, help="prepare the six official AC2 policy releases")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", help="Hub repository the bundle will be uploaded to (for its model card)")
    parser.add_argument("--owner", default="ZibinDong", help="Hub owner for the six-policy release")
    parser.add_argument(
        "--tokenizer-catalog", type=Path,
        help="migrate an AC2 policy to an identical variant in this local catalog and bundle it",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="afterwards rebuild the action tokenizer from the bundle (may download it)",
    )
    args = parser.parse_args()
    if args.release_root is not None:
        if args.tokenizer_catalog is None or args.repo_id is not None:
            parser.error("--release-root requires --tokenizer-catalog and uses --owner, not --repo-id")
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {"format_version": 1, "policies": []}
        for dataset in ("libero", "simplerenv", "droid"):
            for order in ("first", "second"):
                source = args.release_root / f"{dataset}-actioncodec2-{order}-order"
                policy = json.loads((source / "policy_config.json").read_text())
                repo_id = f"{args.owner}/{policy_repo_name(policy)}"
                directory = repo_id.split("/")[-1]
                bundle = export_checkpoint(source, args.output / directory, repo_id,
                                           tokenizer_catalog=args.tokenizer_catalog)
                verify_bundle(bundle)
                policy = json.loads((bundle / "policy_config.json").read_text())
                manifest["policies"].append({
                    "repo_id": repo_id, "directory": directory,
                    "variant": policy["tokenizer"]["variant"],
                    "variant_fingerprint": policy["tokenizer"]["variant_fingerprint"],
                    "files": audit_public_files(bundle),
                })
                print(f"prepared: {repo_id}", flush=True)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return
    bundle = export_checkpoint(args.checkpoint, args.output, args.repo_id,
                               tokenizer_catalog=args.tokenizer_catalog)
    print(bundle)
    if args.verify:
        verify_bundle(bundle)


if __name__ == "__main__":
    main()
