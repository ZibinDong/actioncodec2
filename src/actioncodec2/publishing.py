"""Prepare local assets for publication."""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tokenize


def audit_public_files(directory: str | Path) -> dict[str, str]:
    """Check the exact upload files and return their SHA-256 inventory."""
    import struct

    root = Path(directory)
    forbidden = {
        "run_config.json", "optimizer.pt", "optimizer_state.pt", "SOURCE.json",
        "fit_report.json", "train_report.json", "AGENTS.md", "CLAUDE.md", "CODEX.md",
        "GEMINI.md", "SKILL.md", "__pycache__", ".git", ".cache", "wandb", "swanlog",
        ".pi", ".claude", ".codex", ".agents", ".cursor", ".opencode", ".idea", ".vscode",
    }
    private = re.compile(
        r"/efm-vepfs/|/root/|/home/|/tmp/|hf_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16}|"
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    )
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or any(part in forbidden or part.startswith(".env") for part in relative.parts):
            raise ValueError(f"unpublishable file: {relative.as_posix()}")
        if not path.is_file():
            continue
        if path.suffix in {".log", ".jsonl", ".pyc", ".so", ".pt", ".pth", ".bin"}:
            raise ValueError(f"unpublishable file: {relative.as_posix()}")
        if path.suffix == ".safetensors":
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                metadata = json.loads(stream.read(length))
            text = json.dumps(metadata.get("__metadata__", {}))
        elif (path.name in {"LICENSE", "NOTICE"} or path.suffix in
              {".json", ".yaml", ".yml", ".md", ".py", ".cpp", ".hpp", ".toml", ".txt", ".jinja"}):
            text = path.read_text(encoding="utf-8")
        else:
            text = ""
        if private.search(text):
            raise ValueError(f"private path or credential in {relative.as_posix()}")
        with path.open("rb") as stream:
            files[relative.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return files


def prepare_public_assets(directory: str | Path) -> None:
    """Remove private metadata and explanatory text from copied release assets."""
    root = Path(directory)
    for path in sorted(root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(path)
    for pattern in ("*.pyc", "*.log", "*.jsonl"):
        for path in root.rglob(pattern):
            path.unlink()
    for name in ("fit_report.json", "qc_events.parquet", "train_report.json"):
        for path in root.rglob(name):
            path.unlink()
    asset_names = {
        "tokenizer_action_stats": "action_stats",
        "tokenizer_normalizer_path": "normalizer_path",
        "tokenizer_calibration_path": "calibration_path",
        "tokenizer_statistics_path": "statistics_path",
        "model_proprio_stats": "proprio_stats",
    }
    kinds = {path.resolve(): asset_names[path.name] for path in root.rglob("*")
             if path.is_file() and path.name in asset_names}
    checkpoints = {path.resolve() for path in root.rglob("tokenizer_artifact") if path.is_file()}
    paths = [*sorted(root.rglob("*.json"), key=lambda path: path.name != "policy_config.json"), *kinds]
    seen = set()
    for path in paths:
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        original = dict(payload)
        if path.name == "policy_config.json" and payload.get("format_version") == 1:
            tokenizer = dict(payload["tokenizer"])
            for key in ("artifact", "processor", "source", "action_stats", "normalizer_path",
                        "statistics_path", "calibration_path"):
                value = tokenizer.get(key)
                if isinstance(value, str) and value.startswith(("/", "~", "./", "../")):
                    asset = (path.parent / Path(value).expanduser()).resolve()
                    if not asset.is_relative_to(path.parent.resolve()):
                        raise ValueError(f"published tokenizer.{key} points outside its bundle")
                    tokenizer[key] = "./" + asset.relative_to(path.parent.resolve()).as_posix()
                    if key in {"action_stats", "normalizer_path", "calibration_path", "statistics_path"} and asset.is_file():
                        kinds[asset] = key
                        paths.append(asset)
                    if key == "artifact" and tokenizer.get("name") == "x-tokenizer":
                        if asset.is_file():
                            checkpoints.add(asset)
                        elif asset.is_dir():
                            checkpoints.update(asset / name for name in ("xtokenizer.pth", "best.pth", "latest.pth")
                                               if (asset / name).is_file())
            payload["tokenizer"] = tokenizer
        if payload.get("schema") == "actioncodec2.x-tokenizer-output-calibration/2":
            payload = {key: payload[key] for key in (
                "schema", "horizon", "weight", "bias", "gripper_threshold",
                "clip_low", "clip_high", "feature_type",
            ) if key in payload}
        elif kinds.get(path.resolve()) == "normalizer_path":
            payload = {key: payload[key] for key in ("low", "high")}
        elif kinds.get(path.resolve()) == "proprio_stats":
            payload = {key: payload[key] for key in (
                "train_mean", "train_std", "q01", "q99", "quantile_indices",
            ) if key in payload}
        elif kinds.get(path.resolve()) == "action_stats":
            bounds = {"q001", "q999", "q01", "q99", "low", "high", "gripper_threshold"}
            payload = {
                key: ({name: item for name, item in value.items() if name in bounds}
                      if isinstance(value, dict) else value)
                for key, value in payload.items()
                if key in bounds or (
                    isinstance(value, dict) and any(
                        low in value and high in value
                        for low, high in (("q001", "q999"), ("q01", "q99"), ("low", "high"))
                    )
                )
            }
        for key in ("_name_or_path", "name_or_path"):
            payload.pop(key, None)
        if path.name in {"tokenizer_config.json", "generation_config.json", "special_tokens_map.json"}:
            for key in ("tokenizer_file", "vocab_file", "merges_file", "special_tokens_map_file",
                        "added_tokens_file"):
                if isinstance(payload.get(key), str) and payload[key].startswith(("/", "~")):
                    payload.pop(key)
        if payload != original:
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if checkpoints:
        import torch
        import zipfile

        for path in checkpoints:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True,
                                    mmap=zipfile.is_zipfile(path))
            if not isinstance(checkpoint, dict) or not {
                "model_state_dict", "architecture", "data_spec"
            } <= checkpoint.keys():
                raise ValueError("expected an X-Tokenizer inference checkpoint")
            if "finetune" not in checkpoint:
                continue
            json.dumps(checkpoint["finetune"])
            del checkpoint["finetune"]
            temporary = path.with_name(f".{path.name}.public")
            torch.save(checkpoint, temporary)
            temporary.replace(path)

    for entry in root.rglob("processing_actioncodec2.py"):
        runtime = entry.parent / "runtime"
        if not runtime.is_dir():
            continue
        for path in [entry, *runtime.rglob("*.py")]:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            lines = source.splitlines(keepends=True)
            offsets = [0]
            for line in lines:
                offsets.append(offsets[-1] + len(line.encode("utf-8")))
            encoded = source.encode("utf-8")
            edits = []
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                doc = ast.get_docstring(node)
                if not doc:
                    continue
                if any(marker in doc.lower() for marker in ("copyright", "spdx", "licensed under")):
                    continue
                expression = node.body[0]
                if getattr(node, "name", "").startswith("_"):
                    replacement = "pass" if len(node.body) == 1 else ""
                elif "\n\n" in doc:
                    replacement = json.dumps(doc.split("\n\n")[0], ensure_ascii=False)
                else:
                    continue
                edits.append((offsets[expression.lineno - 1] + expression.col_offset,
                              offsets[expression.end_lineno - 1] + expression.end_col_offset,
                              replacement.encode("utf-8")))
            for start, end, replacement in sorted(edits, reverse=True):
                encoded = encoded[:start] + replacement + encoded[end:]
            source = encoded.decode("utf-8")
            lines = source.splitlines(keepends=True)
            blocks: list[list[tokenize.TokenInfo]] = []
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type != tokenize.COMMENT:
                    continue
                if blocks and token.start[0] == blocks[-1][-1].start[0] + 1:
                    blocks[-1].append(token)
                else:
                    blocks.append([token])
            for block in blocks:
                text = " ".join(token.string for token in block).lower()
                if any(marker in text for marker in (
                    "spdx", "copyright", "license", "adapted from", "derived from",
                    "noqa", "type: ignore", "pragma", "coding:", "#!",
                )):
                    continue
                for token in block:
                    row, column = token.start
                    lines[row - 1] = lines[row - 1][:column].rstrip() + "\n"
            cleaned = re.sub(r"\n{4,}", "\n\n\n", "".join(lines))
            ast.parse(cleaned)
            if cleaned != path.read_text(encoding="utf-8"):
                path.write_text(cleaned, encoding="utf-8")

    for path in root.rglob("README.md"):
        source = path.read_text(encoding="utf-8")
        cleaned = source.replace(
            "Trained on curated data with lower-quality datasets and most stationary-arm segments removed.",
            "Extended ActionCodec2 tokenizer weights.",
        )
        if cleaned != source:
            path.write_text(cleaned, encoding="utf-8")
