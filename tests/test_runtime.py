"""Tokenizer, policy export, Hugging Face model, and serving contracts."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.sync.client import connect
from websockets.sync.server import serve

from actioncodec2.tokenizers.actioncodec import ActionCodecNormalizer
from actioncodec2.tokenizers.factory import check_tokenizer_settings
from actioncodec2.tokenizers.fast import FastGrammar, _bytes_to_unicode_inverse
from actioncodec2.vla.runtime.run import load_runtime_config, tokenizer_settings
from actioncodec2.vla.runtime.server import MicroBatcher, make_handler, pack_array
from actioncodec2.vla.training.config import DataConfig, ModelConfig, RunConfig, TrainConfig

ROOT = Path(__file__).resolve().parents[1]
AC2_ARTIFACT_ROOT = Path(os.environ.get("ACTIONCODEC2_TEST_ARTIFACT_ROOT", "/nonexistent"))
HORIZON, ACTION_DIM = 4, 2


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLIENT = _load_script("benchmark_client")
POLICY_EXPORT = _load_script("export_policy_bundle")
TOKENIZER_EXPORT = _load_script("export_tokenizer_bundle")
PUBLISH = _load_script("publish_huggingface")


CATALOG_VARIANTS = [
    "paper-o1-5mm-4k-2bin-1p2ms", "paper-o2-5mm-4k-2bin-1p2ms",
    *(f"extended-o{order}-{precision}-{size}k-2bin-1p2ms"
      for order in (1, 2) for precision in ("1mm", "5mm", "10mm")
      for size in (1, 2, 4, 8, 16) if precision != "1mm" or size != 1),
]


@pytest.fixture(scope="module")
def merged_catalog(tmp_path_factory):
    source = _artifact("first_order").resolve().parent
    if not (source / "manifest.json").is_file():
        pytest.skip("the full tokenizer release catalog is not installed")
    output = tmp_path_factory.mktemp("catalog") / "ActionCodec2"
    TOKENIZER_EXPORT.export_catalog(source, output, repo_id="ZibinDong/ActionCodec2")
    return source, output


@pytest.mark.slow
@pytest.mark.parametrize("variant", CATALOG_VARIANTS)
def test_catalog_variants_preserve_tokens_decoding_grammar_and_offline_save(
    variant, merged_catalog, monkeypatch, tmp_path,
):
    from transformers import AutoProcessor
    from actioncodec2.tokenizers.hub_catalog import asset_inventory
    from actioncodec2.core.catalog import ActionCodec2, VARIANT_ALIASES

    source, root = merged_catalog
    catalog = json.loads((root / "variants.json").read_text())["variants"]
    record = catalog[variant]
    old = source / record["legacy_repo_id"].split("/")[-1]
    assert asset_inventory(root / "variants" / variant) == record["files"]
    merged = AutoProcessor.from_pretrained(
        root, variant=variant, trust_remote_code=True, local_files_only=True,
    )
    legacy_name = next(old for old, new in VARIANT_ALIASES.items() if new == variant)
    alias_codec = ActionCodec2.from_pretrained(root, variant=legacy_name)
    assert alias_codec.variant == variant
    assert alias_codec.variant_fingerprint == merged.variant_fingerprint
    # Load the frozen legacy runtime directly: its Hub entry can select the installed package.
    package_name = f"_legacy_ac2_test_{abs(hash(str(old)))}"
    spec = importlib.util.spec_from_file_location(
        package_name, old / "runtime/__init__.py", submodule_search_locations=[str(old / "runtime")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    importlib.import_module(f"{package_name}.hub_acceleration").install(package_name)
    original = importlib.import_module(f"{package_name}.processing_actioncodec2").ActionCodec2.from_pretrained(old)
    assert merged.variant == variant and merged.variant_fingerprint == record["fingerprint"]
    for name, profile in merged.action_tokenizer.profiles.items():
        assert profile.processor.artifact_fingerprint == original.action_tokenizer.profiles[name].processor.artifact_fingerprint
    layouts = [("single_eef_delta", 7), ("dual_eef_delta", 14), ("single_eef_abs", 7)]
    if "joint" in record["profiles"]:
        layouts.extend([("single_joint6_delta", 7), ("single_joint7_delta", 8)])
    else:
        with pytest.raises(KeyError):
            merged.for_action_space("single_joint7_delta")
    rng = np.random.default_rng(123)
    for layout, width in layouts:
        new_codec, old_codec = merged.for_action_space(layout), original.for_action_space(layout)
        assert new_codec.variant == variant
        actions = rng.normal(0, 0.002, (2, 12, width)).astype(np.float32)
        for component in new_codec.action_space.components:
            if component.gripper_dim is not None:
                actions[..., component.gripper_dim] = rng.choice([0.0, 0.6, 0.8, 1.0], (2, 12))
        kwargs = {}
        if new_codec.requires_previous_first_order:
            kwargs["previous_first_order"] = {
                k: rng.normal(0, 0.001, v.shape).astype(np.float32)
                for k, v in new_codec.first_order_template(2).items()
            }
        if new_codec.action_space.input == "absolute":
            kwargs["current_state"] = np.zeros((2, width), np.float32)
        for rate in (15, 20):
            expected = old_codec.encode(actions, fps=rate, **kwargs)
            expected_decoded = old_codec.decode(expected, fps=rate, **kwargs).numpy()
            tokens = new_codec.encode(actions, fps=rate, **kwargs)
            assert alias_codec.for_action_space(layout).encode(actions, fps=rate, **kwargs) == tokens
            dual = len(new_codec.action_space.components) > 1
            if dual:
                router = new_codec.action_tokenizer
                steps = new_codec.grammar(12, fps=rate).horizon
                for row, previous in zip(tokens, expected, strict=True):
                    streams = router._split_stream(row, preset=new_codec.action_space, steps=steps)
                    segments = router._segments(new_codec.action_space)
                    assert [t + segment.profile.spec.token_offset for stream, segment in zip(streams, segments, strict=True) for t in stream] == previous
                assert new_codec.action_space.token_order == "time_major"
            else:
                assert tokens == expected
            np.testing.assert_array_equal(new_codec.decode(tokens, fps=rate, **kwargs).numpy(),
                                          expected_decoded)
            grammar, old_grammar = new_codec.grammar(12, fps=rate), old_codec.grammar(12, fps=rate)
            assert all(grammar.is_complete(row) for row in tokens)
            for end in (0, 1, len(tokens[0]) // 2, len(tokens[0])):
                prefix = tokens[0][:end]
                if not dual:
                    np.testing.assert_array_equal(grammar.next_token_mask(prefix),
                                                  old_grammar.next_token_mask(prefix))
                elif end < len(tokens[0]):
                    assert grammar.next_token_mask(prefix)[tokens[0][end]]
            monkeypatch.setenv("ACTIONCODEC2_NO_NATIVE", "1")
            python_expected = old_codec.encode(actions, fps=rate, **kwargs)
            assert python_expected == expected
            assert new_codec.encode(actions, fps=rate, **kwargs) == tokens
            python_decoded = old_codec.decode(expected, fps=rate, **kwargs).numpy()
            np.testing.assert_array_equal(new_codec.decode(tokens, fps=rate, **kwargs).numpy(),
                                          python_decoded)
            np.testing.assert_allclose(python_decoded, expected_decoded, rtol=1e-6, atol=1e-7)
            if not dual:
                np.testing.assert_array_equal(grammar.next_token_mask([]), old_grammar.next_token_mask([]))
            monkeypatch.delenv("ACTIONCODEC2_NO_NATIVE")
            if new_codec.requires_previous_first_order:
                for executed in (None, 3):
                    result = new_codec.final_first_order(
                        tokens, previous_first_order=kwargs["previous_first_order"], executed_steps=executed,
                    )
                    previous = old_codec.final_first_order(
                        expected, previous_first_order=kwargs["previous_first_order"], executed_steps=executed,
                    )
                    for key in result:
                        np.testing.assert_array_equal(result[key], previous[key])
    saved = tmp_path / "saved"
    merged.for_action_space("dual_eef_delta").save_pretrained(saved)
    assert not (saved / "variants").exists() and not (saved / "variants.json").exists()
    assert (saved / "artifact_manifest.json").is_file()
    assert (saved / "actioncodec2_kernels/grammar.cpp").is_file()
    for name, digest in asset_inventory(root / "variants" / variant).items():
        if name.endswith(".npz"):
            with np.load(saved / name, allow_pickle=False) as actual, np.load(
                root / "variants" / variant / name, allow_pickle=False
            ) as expected:
                assert actual.files == expected.files
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], expected[key])
        elif name.endswith(".json") and name != "processor_config.json":
            assert json.loads((saved / name).read_text()) == json.loads(
                (root / "variants" / variant / name).read_text()
            )
        elif name != "processor_config.json":
            assert asset_inventory(saved)[name] == digest
    restored = AutoProcessor.from_pretrained(saved, trust_remote_code=True, local_files_only=True)
    assert restored.variant == variant and restored.action_space.name == "dual_eef_delta"
    actions = np.zeros((2, 12, 14), np.float32)
    kwargs = ({"previous_first_order": restored.first_order_template(2)}
              if restored.requires_previous_first_order else {})
    assert restored.encode(actions, fps=15, **kwargs) == merged.for_action_space(
        "dual_eef_delta"
    ).encode(actions, fps=15, **kwargs)


@pytest.mark.slow
def test_catalog_rejects_wrong_variants_corrupt_assets_and_survives_isolated_hub_cache(
    merged_catalog, tmp_path,
):
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from actioncodec2.tokenizers import build_tokenizer
    from actioncodec2.tokenizers.hub_catalog import ActionCodec2

    _, source = merged_catalog
    refreshed = tmp_path / "refreshed"
    TOKENIZER_EXPORT.export_bundle(source, refreshed)
    assert (source / "variants.json").read_bytes() == (refreshed / "variants.json").read_bytes()
    codec = AutoProcessor.from_pretrained(refreshed, variant="paper-o2-5mm-4k-2bin-1p2ms", trust_remote_code=True,
                                         local_files_only=True)
    assert codec.variant == "paper-o2-5mm-4k-2bin-1p2ms"
    root = tmp_path / "catalog"
    shutil.copytree(source, root)
    for bad in (None, "missing", "../paper-o1-5mm-4k-2bin-1p2ms", "/paper-o1-5mm-4k-2bin-1p2ms", True):
        with pytest.raises(ValueError, match="variant"):
            AutoProcessor.from_pretrained(root, variant=bad, trust_remote_code=True, local_files_only=True)
    with pytest.raises(ValueError, match="subfolder"):
        ActionCodec2.from_pretrained(root, variant="paper-o1-5mm-4k-2bin-1p2ms", subfolder="variants/paper-o1-5mm-4k-2bin-1p2ms",
                                    local_files_only=True)
    with pytest.raises(ValueError, match="fingerprint"):
        AutoProcessor.from_pretrained(root, variant="paper-o1-5mm-4k-2bin-1p2ms", variant_fingerprint="0" * 64,
                                     trust_remote_code=True, local_files_only=True)
    data = root / "variants/paper-o1-5mm-4k-2bin-1p2ms/profiles/eef/bpe.json"
    original = data.read_bytes()
    data.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="fingerprint"):
        AutoProcessor.from_pretrained(root, variant="paper-o1-5mm-4k-2bin-1p2ms", trust_remote_code=True, local_files_only=True)
    data.write_bytes(original)
    for variant in ("paper-o1-5mm-4k-2bin-1p2ms", "paper-o2-5mm-4k-2bin-1p2ms"):
        settings = {"name": "actioncodec2", "artifact": str(root), "variant": variant,
                    "time_horizon": 12, "fps": 15}
        adapter = build_tokenizer(settings)
        assert adapter.to_json()["variant"] == variant
        boundary = np.zeros((2, 7), np.float32) if adapter.requires_previous_action else None
        actions = np.zeros((2, 12, 7), np.float32)
        assert adapter.encode_batch(actions, boundary) == build_tokenizer(
            adapter.to_json()
        ).encode_batch(actions, boundary)
    codec = AutoProcessor.from_pretrained(root, variant="paper-o2-5mm-4k-2bin-1p2ms", trust_remote_code=True,
                                         local_files_only=True)
    original_fingerprint = codec.variant_fingerprint
    inventory = json.loads((root / "variants.json").read_text())
    inventory["variants"]["paper-o2-5mm-4k-2bin-1p2ms"]["files"]["../../outside"] = "0" * 64
    (root / "variants.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="fingerprint"):
        ActionCodec2.from_pretrained(root, variant="paper-o2-5mm-4k-2bin-1p2ms", local_files_only=True)
    shutil.copyfile(source / "variants.json", root / "variants.json")
    with pytest.raises(ValueError, match="separate directory"):
        codec.save_pretrained(root)
    saved = tmp_path / "saved"
    codec.save_pretrained(saved)
    damaged = saved / "profiles/eef/bpe.json"
    damaged.write_bytes(damaged.read_bytes() + b" ")
    with pytest.raises(ValueError, match="fingerprint"):
        ActionCodec2.from_pretrained(saved, local_files_only=True)
    codec.save_pretrained(saved)
    assert codec.variant_fingerprint == original_fingerprint
    with pytest.raises(ValueError, match="different variant"):
        AutoProcessor.from_pretrained(saved, variant="paper-o1-5mm-4k-2bin-1p2ms", trust_remote_code=True,
                                     local_files_only=True)
    for variant, path in (("paper-o1-5mm-4k-2bin-1p2ms", root / "variants/paper-o1-5mm-4k-2bin-1p2ms"), (None, saved)):
        cache = tmp_path / (variant or "standalone")
        repository = cache / "models--example--actioncodec2"
        commit = "a" * 40
        snapshot = repository / "snapshots" / commit
        snapshot.mkdir(parents=True)
        (repository / "refs").mkdir()
        (repository / "refs/main").write_text(commit)
        for file in root.iterdir() if variant else saved.iterdir():
            if file.name == "variants":
                continue
            if file.is_file():
                shutil.copyfile(file, snapshot / file.name)
            elif file.is_dir():
                shutil.copytree(file, snapshot / file.name)
        if variant:
            shutil.copytree(path, snapshot / "variants" / variant)
        import hashlib

        blobs = repository / "blobs"
        blobs.mkdir()
        for cached in snapshot.rglob("*"):
            if not cached.is_file():
                continue
            payload = cached.read_bytes()
            blob = blobs / hashlib.sha256(payload).hexdigest()
            if not blob.exists():
                blob.write_bytes(payload)
            cached.unlink()
            cached.symlink_to(os.path.relpath(blob, cached.parent))
        assert Path(snapshot_download("example/actioncodec2", cache_dir=cache,
                                      local_files_only=True)) == snapshot
        shutil.rmtree(root if variant else saved)
        script = '''
import sys
from pathlib import Path
import numpy as np
from transformers import AutoProcessor
codec = AutoProcessor.from_pretrained(
    "example/actioncodec2", cache_dir=sys.argv[1], variant=sys.argv[2] or None,
    revision="a" * 40, local_files_only=True, trust_remote_code=True,
)
assert codec.variant == (sys.argv[2] or "paper-o2-5mm-4k-2bin-1p2ms")
values = np.zeros((1, 12, 7), np.float32)
kwargs = {"previous_first_order": codec.first_order_template(1)} if codec.requires_previous_first_order else {}
tokens = codec.encode(values, fps=15, **kwargs)
assert codec.decode(tokens, fps=15, **kwargs).shape == values.shape
codec.save_pretrained(Path(sys.argv[1]) / "exported")
'''
        result = subprocess.run(
            [sys.executable, "-c", script, str(cache), variant or ""],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "HF_MODULES_CACHE": str(cache / "modules")},
            text=True, capture_output=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["paper-o1-5mm-4k-2bin-1p2ms", "paper-o2-5mm-4k-2bin-1p2ms", "extended-o2-1mm-2k-2bin-1p2ms"])
def test_catalog_policy_export_bundles_only_the_selected_variant(merged_catalog, tmp_path, variant):
    from actioncodec2.vla.runtime.run import build_tokenizer_for

    _, root = merged_catalog
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"fixture")
    (source / "tokenizer.json").write_text("{}")
    (source / "config.json").write_text(json.dumps({"model_type": "neo_one_step_vla"}))
    (source / "policy_config.json").write_text(json.dumps({
        "action_vocab_size": 4096, "action_token_begin": 248077, "total_vocab_size": 252189,
        "execution_stride": 2,
    }))
    config = RunConfig(
        ModelConfig(path="/old/base.pt", tokenizer_path="/old/text", proprio_mode="none"),
        {"name": "actioncodec2", "artifact": str(root), "variant": variant},
        DataConfig(roots=["/old/data"], horizon=12, control_frequency_hz=15),
        TrainConfig(output_dir="/old/train"),
    )
    original = build_tokenizer_for(config)
    assert config.tokenizer["variant_fingerprint"] == original.variant_fingerprint
    config.save(source / "run_config.json")
    bundle = POLICY_EXPORT.export_checkpoint(source, tmp_path / "bundle")
    loaded = load_runtime_config(bundle)
    settings = tokenizer_settings(loaded)
    copied = bundle / settings["artifact"]
    assert not (copied / "variants").exists() and not (copied / "variants.json").exists()
    assert (copied / "artifact_manifest.json").is_file()
    assert settings["variant"] == variant
    assert settings["variant_fingerprint"] == original.variant_fingerprint
    restored = build_tokenizer_for(loaded)
    actions = np.zeros((2, 12, 7), np.float32)
    boundary = np.zeros((2, 7), np.float32) if original.requires_previous_action else None
    tokens = original.encode_batch(actions, boundary)
    assert tokens == restored.encode_batch(actions, boundary)
    np.testing.assert_array_equal(original.decode_batch(tokens, boundary),
                                  restored.decode_batch(tokens, boundary))
    assert "corpusfit" not in POLICY_EXPORT.policy_label(json.loads((bundle / "policy_config.json").read_text()))
    refreshed = POLICY_EXPORT.export_checkpoint(bundle, tmp_path / "refreshed")
    refreshed_tokenizer = build_tokenizer_for(load_runtime_config(refreshed))
    assert refreshed_tokenizer.variant == variant
    assert refreshed_tokenizer.encode_batch(actions, boundary) == tokens
    assert (refreshed / "model.safetensors").read_bytes() == b"fixture"
    assert "trust_remote_code=True" in (refreshed / "README.md").read_text()


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["paper-o1-5mm-4k-2bin-1p2ms", "paper-o2-5mm-4k-2bin-1p2ms"])
def test_policy_migration_checks_legacy_assets_and_bundles_current_runtime(merged_catalog, tmp_path, variant):
    from actioncodec2.tokenizers import build_tokenizer
    from actioncodec2.vla.runtime.run import build_tokenizer_for

    source_root, catalog = merged_catalog
    record = json.loads((catalog / "variants.json").read_text())["variants"][variant]
    legacy = tmp_path / record["legacy_repo_id"].split("/")[-1]
    shutil.copytree(source_root / legacy.name, legacy)
    (legacy / "runtime/hub_acceleration.py").unlink(missing_ok=True)
    settings = {"name": "actioncodec2", "artifact": str(legacy), "time_horizon": 12, "fps": 15}
    original = build_tokenizer(settings)
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"fixture")
    (source / "tokenizer.json").write_text("{}")
    (source / "config.json").write_text(json.dumps({"model_type": "g05_action_policy"}))
    (source / "policy_config.json").write_text(json.dumps({
        "action_vocab_size": 4096, "action_token_begin": 248077, "total_vocab_size": 252189,
    }))
    config = RunConfig(
        ModelConfig(proprio_mode="none"), settings,
        DataConfig(roots=[], horizon=12, control_frequency_hz=15), TrainConfig(output_dir="/old/train"),
    )
    config.save(source / "run_config.json")
    bundle = POLICY_EXPORT.export_checkpoint(source, tmp_path / "bundle", tokenizer_catalog=catalog)
    restored = build_tokenizer_for(load_runtime_config(bundle))
    assert restored.variant == variant and restored.variant_fingerprint == record["fingerprint"]
    assert restored.revision is None
    assert (bundle / "assets/tokenizer_artifact/runtime/backends/actioncodec2/physical_codec.py").is_file()
    actions = np.zeros((2, 12, 7), np.float32)
    boundary = np.zeros((2, 7), np.float32) if original.requires_previous_action else None
    assert restored.encode_batch(actions, boundary) == original.encode_batch(actions, boundary)
    asset = legacy / "profiles/eef/bpe.json"
    data = json.loads(asset.read_text())
    data["different_trained_assets"] = True
    asset.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changes trained assets"):
        POLICY_EXPORT.export_checkpoint(source, tmp_path / "wrong", tokenizer_catalog=catalog)
    assert not (tmp_path / "wrong").exists()


def test_g05_defaults_preserve_existing_checkpoint_vocabulary():
    from actioncodec2.vla.g05_hf.configuration_g05 import G05Config

    config = G05Config()
    assert (
        config.action_token_begin_idx,
        config.eov_token_id,
        config.state_token_id,
        config.text_vocab_size,
    ) == (248_077, 252_176, 252_177, 252_189)


def test_training_recipes_resolve_data_and_tokenizer_contracts(tmp_path):
    recipes = [ROOT / "configs/quickstart.yaml", *sorted((ROOT / "configs/train").glob("*/*.yaml"))]
    for recipe in recipes:
        if recipe.name == "base.yaml":
            continue
        config = RunConfig.load(recipe, data_root=tmp_path)
        check_tokenizer_settings(config.tokenizer)
        assert RunConfig.from_dict(config.to_json()).to_json() == config.to_json()
        assert config.data.roots and all(Path(p).is_relative_to(tmp_path) for p in config.data.roots)
        for section, keys in RunConfig.PATH_KEYS.items():
            values = config.tokenizer if section == "tokenizer" else vars(getattr(config, section))
            for key in keys:
                value = values.get(key)
                if isinstance(value, str) and Path(value).is_absolute():
                    assert Path(value).is_file(), (recipe, key)
        settings = tokenizer_settings(config)
        assert settings["time_horizon"] == config.data.horizon
        assert settings["gripper_open_is_high"] == config.tokenizer.get(
            "gripper_open_is_high", config.data.gripper_open_is_high
        )
    with pytest.raises(ValueError, match="unknown tokenizer settings"):
        check_tokenizer_settings({"name": "fast", "gripper_open_is_hight": False})


@pytest.mark.parametrize("kind", ["fast", "actioncodec"])
@pytest.mark.parametrize("custom_hub", [False, True])
def test_exported_policy_preserves_inference_assets_and_omits_training_paths(tmp_path, kind, custom_hub):
    bounds = {"low": [-1.0] * 7, "high": [1.0] * 7}
    releases = {
        "first": ("ZibinDong/ActionCodec2-1st-order", "511a2cf7e5801f058ff24d3bed9b1120cee26671"),
        "second": ("ZibinDong/ActionCodec2-2nd-order", "1aaa5579b0a331f63847556fd39543bf171a0359"),
    }
    if kind in releases:
        repo, revision = releases[kind]
        tokenizer = {"name": "actioncodec2", "artifact": repo,
                     "revision": revision, "action_space": "delta_eef"}
        if kind == "second":
            tokenizer.update(previous_action_resampling="constant_velocity",
                             resampling_alignment="interval_end")
    elif kind == "fast":
        tokenizer = {"name": "fast", "processor": POLICY_EXPORT.DEFAULT_FAST, "normalizer": bounds}
    else:
        tokenizer = {"name": "actioncodec", "artifact": POLICY_EXPORT.DEFAULT_ACTIONCODEC, "action_clip": bounds}
    field = "processor" if kind == "fast" else "artifact"
    if custom_hub:
        tokenizer[field] = f"example/custom-{kind}"
        tokenizer["revision"] = "a" * 40
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"fixture")
    (source / "tokenizer.json").write_text("{}")
    (source / "config.json").write_text(json.dumps({
        "model_type": "neo_one_step_vla", "_name_or_path": "/private/checkpoint",
    }))
    (source / "tokenizer_config.json").write_text(json.dumps({
        "name_or_path": "/private/tokenizer", "tokenizer_file": "/private/tokenizer.json",
    }))
    saved = {"action_vocab_size": 4, "action_token_begin": 8, "total_vocab_size": 16,
             "execution_stride": 2}
    (source / "policy_config.json").write_text(json.dumps(saved))
    config = RunConfig(
        ModelConfig(path="/old/base.pt", tokenizer_path="/old/text", proprio_mode="none"),
        tokenizer, DataConfig(roots=["/old/data"]), TrainConfig(output_dir="/old/train"),
    )
    config.save(source / "run_config.json")
    assert load_runtime_config(source).to_json() == config.to_json()
    original = (source / "run_config.json").read_bytes()
    bundle = POLICY_EXPORT.export_checkpoint(source, tmp_path / "bundle")
    loaded = load_runtime_config(bundle)
    assert loaded.model.path == loaded.model.tokenizer_path == str(bundle)
    assert tokenizer_settings(loaded) == tokenizer_settings(config)
    assert loaded.data.execution_stride == saved["execution_stride"]
    assert loaded.data.roots == [] and not hasattr(loaded, "train")
    assert not (bundle / "run_config.json").exists() and not (bundle / "assets").exists()
    assert "/old/" not in (bundle / "policy_config.json").read_text()
    assert "/private/" not in (bundle / "config.json").read_text()
    assert "/private/" not in (bundle / "tokenizer_config.json").read_text()
    assert (source / "run_config.json").read_bytes() == original
    for name in ("model.safetensors", "LICENSE", "NOTICE"):
        assert (bundle / name).is_file()

    artifact = tmp_path / "corpus-tokenizer"
    local_files = ("AGENTS.md", ".pi/session.jsonl", "runtime/.env", "runtime/CLAUDE.md")
    for name in (*local_files, "config.json"):
        path = artifact / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"custom": true}')
    (artifact / "processing_actioncodec2.py").write_text('"""Tokenizer entry point."""\n')
    helper = artifact / "runtime/operations.py"
    helper.write_text(
        '# SPDX-License-Identifier: Apache-2.0\n\n'
        '# Internal experiment notes.\n'
        'def scale(value):\n'
        '    """Scale a value.\n\n    Internal experiment: seed 123, measured error 0.1.\n    """\n'
        '    return value * 2\n'
    )
    calibration = {
        "schema": "actioncodec2.x-tokenizer-output-calibration/2", "horizon": 1,
        "weight": [[[1.0]]], "bias": [[0.0]], "gripper_threshold": [0.5],
        "clip_low": [-1.0], "clip_high": [1.0], "feature_type": "linear",
        "fit": {"seed": 123}, "selection": {"metrics": {"error": 0.1}},
    }
    (artifact / "tokenizer_calibration_path").write_text(json.dumps(calibration))
    config.tokenizer[field] = str(artifact)
    if kind == "fast":
        config.tokenizer["source"] = config.tokenizer.pop("processor")
    config.save(source / "run_config.json")
    exported = POLICY_EXPORT.export_checkpoint(source, tmp_path / "bundled-policy")
    settings = tokenizer_settings(load_runtime_config(exported))
    copied = exported / settings[field]
    assert json.loads((copied / "config.json").read_text()) == {"custom": True}
    assert settings.get("revision") == config.tokenizer.get("revision")
    if kind == "fast":
        assert "source" not in settings
    assert all(not (copied / name).exists() for name in local_files)
    published = (copied / "runtime/operations.py").read_text()
    assert "SPDX-License-Identifier: Apache-2.0" in published
    assert "Internal experiment" not in published and "seed 123" not in published
    namespace = {}
    exec(compile(published, "published_operations.py", "exec"), namespace)
    assert namespace["scale"](3) == 6
    cleaned_calibration = json.loads((copied / "tokenizer_calibration_path").read_text())
    assert "fit" not in cleaned_calibration and "selection" not in cleaned_calibration
    assert cleaned_calibration["weight"] == calibration["weight"]
    policy_path = exported / "policy_config.json"
    payload = json.loads(policy_path.read_text())
    payload["tokenizer"][field] = str(copied.resolve())
    policy_path.write_text(json.dumps(payload))
    POLICY_EXPORT.prepare_public_assets(exported)
    assert tokenizer_settings(load_runtime_config(exported)) == settings
    payload["tokenizer"][field] = str(tmp_path / "outside-bundle")
    policy_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="outside its bundle"):
        POLICY_EXPORT.prepare_public_assets(exported)


def _require_actioncodec2_artifact(profile: str) -> None:
    if not (AC2_ARTIFACT_ROOT / profile).is_dir():
        pytest.skip("external ActionCodec2 artifact is not installed")


def _actions():
    values = np.zeros((20, 7), np.float32)
    values[:, 0] = np.linspace(-0.1, 0.1, 20)
    values[:, 6] = 1.0
    return values


@pytest.mark.slow
@pytest.mark.parametrize(
    ("profile", "action_space"),
    [("first_order", "single_eef_delta"), ("first_order", "single_joint7_delta"),
     ("second_order", "single_eef_delta")],
)
def test_actioncodec2_native_path_matches_bundled_runtime(
    profile: str, action_space: str, monkeypatch: pytest.MonkeyPatch
):
    _require_actioncodec2_artifact(profile)
    from actioncodec2.tokenizers import build_tokenizer

    settings = {"name": "actioncodec2", "artifact": str(AC2_ARTIFACT_ROOT / profile),
                "time_horizon": 20, "fps": 20, "target_hz": 15, "action_space": action_space,
                "rotation_multiplication": "left"}
    if profile == "second_order":
        settings.update(previous_action_resampling="constant_velocity",
                        resampling_alignment="interval_end")
    native = build_tokenizer(settings)
    monkeypatch.setenv("ACTIONCODEC2_NO_NATIVE", "1")
    runtime = build_tokenizer(settings)
    monkeypatch.delenv("ACTIONCODEC2_NO_NATIVE")
    assert native._setbpe is not None and runtime._native is None

    rng = np.random.default_rng(0)
    width = native.action_dim - 1
    chunks = np.concatenate((
        np.cumsum(rng.normal(0.0, 0.004, (8, 20, width)), axis=1),
        (rng.random((8, 20, 1)) < 0.5),
    ), axis=-1).astype(np.float32)
    boundary = (
        (rng.normal(0.0, 0.004, (8, width + 1)) * ([1] * width + [0])).astype(np.float32)
        if native.requires_previous_action else None
    )
    np.testing.assert_array_equal(native.canonical_targets(chunks), runtime.canonical_targets(chunks))
    streams = native.encode_batch(chunks, boundary)
    assert streams == runtime.encode_batch(chunks, boundary)
    np.testing.assert_array_equal(
        native.decode_batch(streams, boundary), runtime.decode_batch(streams, boundary)
    )
    assert all(native.grammar.is_complete(row) for row in streams)
    for stream in streams[:2]:
        for index in range(len(stream) + 1):
            prefix = stream[:index]
            monkeypatch.setenv("ACTIONCODEC2_NO_NATIVE", "1")
            expected_mask = runtime.grammar.next_token_mask(prefix)
            monkeypatch.delenv("ACTIONCODEC2_NO_NATIVE")
            np.testing.assert_array_equal(native.grammar.next_token_mask(prefix), expected_mask)
    for bad in ([float(token) + 0.25 for token in streams[0]],
                [str(token) for token in streams[0]], [True], [-1], [native.vocab_size]):
        with pytest.raises(ValueError):
            native.decode_batch([bad], None if boundary is None else boundary[:1])
        assert not native.grammar.is_complete(bad)
        with pytest.raises(ValueError, match="integers"):
            native.grammar.next_token_mask(bad)
    if native.requires_previous_action:
        with pytest.raises(ValueError):
            native.encode_batch(chunks)
        from actioncodec2.vla.data.augmentation import PreviousActionAugmentation

        augmented, boundaries, _ = native.encode_augmented_batch(
            chunks, boundary, np.arange(len(chunks)),
            PreviousActionAugmentation(clean_probability=0.0), rng,
        )
        assert all(native.grammar.is_complete(row) for row in augmented)
        assert boundaries.shape == boundary.shape and np.isfinite(boundaries).all()



def test_libero_x_tokenizer_absolute_conversion_roundtrip():
    from actioncodec2.tokenizers.xtokenizer import (
        absolute_batch_from_libero_chunks,
        libero_actions_from_absolute_matrix,
    )

    actions = np.stack((_actions()[:4], _actions()[4:8]))
    actions[..., 3:6] = [0.02, -0.03, 0.01]
    layout = {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
        "follow_left_gripper": 1,
        "unused": 16,
    }
    absolute = absolute_batch_from_libero_chunks(actions, layout)
    restored = libero_actions_from_absolute_matrix(absolute[:, 1:], absolute[:, 0])
    assert absolute.shape == (len(actions), actions.shape[1] + 1, sum(layout.values()))
    np.testing.assert_allclose(restored, actions, atol=1e-6)


def test_actioncodec_gripper_polarity_is_reversible() -> None:
    normalizer = ActionCodecNormalizer([-1.0, 0.0], [1.0, 1.0], gripper_open_is_high=False)
    actions = np.array([[0.2, 0.0], [-0.1, 1.0]], dtype=np.float32)
    encoded = normalizer(actions)
    assert encoded[0, -1] > encoded[1, -1]
    np.testing.assert_allclose(normalizer.inverse(encoded), actions)


def _grammar(scale: float = 100.0, min_token: int = -201) -> FastGrammar:
    """A byte-level BPE with one id per byte: id ``b`` is the single byte ``b``."""
    inverse = _bytes_to_unicode_inverse()
    processor = SimpleNamespace(
        scale=scale,
        min_token=min_token,
        bpe_tokenizer=SimpleNamespace(get_vocab=lambda: dict(inverse)),
    )
    tokenizer = SimpleNamespace(
        vocab_size=256, time_horizon=HORIZON, action_dim=ACTION_DIM, processor=processor
    )
    return FastGrammar(tokenizer)


def test_masks_never_allow_a_token_that_cannot_lead_to_a_complete_chunk() -> None:
    """Greedy decoding under the mask always terminates in a complete chunk."""
    grammar = _grammar()
    rng = np.random.default_rng(0)
    for _ in range(20):
        tokens: list[int] = []
        for _ in range(2 * HORIZON * ACTION_DIM):  # a symbol takes at most two bytes here
            mask = grammar.next_token_mask(tokens)
            if not mask.any():
                break
            tokens.append(int(rng.choice(np.flatnonzero(mask))))
        assert grammar.is_complete(tokens)


def test_fast_budget_covers_multibyte_symbols_and_rejects_invalid_ids() -> None:
    from actioncodec2.tokenizers.fast import FastUnconstrainedGrammar

    grammar = _grammar()
    tokens = list(chr(128).encode("utf-8")) * grammar.target_symbols
    for index, token in enumerate(tokens):
        assert grammar.next_token_mask(tokens[:index])[token]
    assert grammar.is_complete(tokens)
    assert len(tokens) == grammar.max_tokens
    unconstrained = FastUnconstrainedGrammar(grammar)
    assert not unconstrained.is_complete(tokens[:grammar.target_symbols])
    assert unconstrained.is_complete(tokens)
    assert not grammar.is_complete([-1])
    assert not grammar.is_complete([grammar.vocab_size])


def test_native_interfaces_reject_invalid_dimensions_and_indices() -> None:
    from actioncodec2.tokenizers.actioncodec2 import _native_kernels

    native = _native_kernels()
    if native is None:
        pytest.skip("native kernels are disabled")
    codec = native.SetBPE([2], 1, [], [])
    for width in (0, -1, 2):
        with pytest.raises(ValueError, match="width"):
            codec.decode_batch([[0]], width, 1)
    for support in (0, 2):
        with pytest.raises(ValueError, match="full_support"):
            codec.decode_batch([[0]], 1, support)
    with pytest.raises(ValueError, match="dimension"):
        codec.encode_batch(np.zeros((1, 2, 2), np.int64))
    for bad in (0.0, 0.25, "0", True, np.bool_(True), -1, 2, 2**100, np.uint64(2**64 - 1)):
        with pytest.raises(ValueError):
            codec.decode_checked_batch([[bad]], 1, 1)
    np.testing.assert_array_equal(codec.decode_checked_batch([[np.int64(1)]], 1, 1), [[[1]]])

    backing = np.array([[1.0, 12345.0]], np.float32)
    layout = [backing[:, :1], np.array([1], np.int32), np.array([1], np.uint8),
              np.array([-1], np.float32), np.array([1], np.float32)]
    state = np.zeros((1, 1), np.float32)
    for cell in (-1, 1):
        with pytest.raises(ValueError, match="cell index"):
            native.physical_decode_batch(
                *layout, np.array([[[cell]]], np.int32), state, state, 1
            )
    decoded = native.physical_decode_batch(
        *layout, np.zeros((1, 1, 1), np.int32), state, state, 1
    )
    np.testing.assert_array_equal(decoded, np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="shapes"):
        native.physical_encode_batch(*layout, np.zeros((1, 1, 2), np.float32), state, state, 1)
    bad_layout = [layout[0], np.array([], np.int32), *layout[2:]]
    with pytest.raises(ValueError, match="vectors"):
        native.physical_decode_batch(
            *bad_layout, np.zeros((1, 1, 1), np.int32), state, state, 1
        )


def test_publication_removes_checkpoint_metadata_without_changing_tensors(tmp_path):
    import torch

    from actioncodec2.publishing import audit_public_files, prepare_public_assets
    from safetensors.torch import save_file

    checkpoint = {
        "format_version": "1.0", "architecture": {"hidden_dim": 4},
        "data_spec": {"dof_dims": {"arm": 7}},
        "model_state_dict": {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                             "counter": torch.tensor(7, dtype=torch.int64)},
        "finetune": {"source_checkpoint": "/private/weights", "train": {"seed": 123}},
    }
    path = tmp_path / "tokenizer_artifact"
    torch.save(checkpoint, path)
    for name in ("fit_report.json", "qc_events.parquet", "train_report.json"):
        (tmp_path / name).write_text("private experiment data")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/private.pyc").write_bytes(b"private cache")
    (tmp_path / "runtime.log").write_text("private runtime paths")
    prepare_public_assets(tmp_path)
    restored = torch.load(path, weights_only=True, mmap=True)
    assert "finetune" not in restored
    assert restored["architecture"] == checkpoint["architecture"]
    assert restored["data_spec"] == checkpoint["data_spec"]
    for name, value in checkpoint["model_state_dict"].items():
        torch.testing.assert_close(restored["model_state_dict"][name], value, rtol=0, atol=0)
    assert all(not (tmp_path / name).exists() for name in (
        "fit_report.json", "qc_events.parquet", "train_report.json", "__pycache__", "runtime.log",
    ))
    inventory = audit_public_files(tmp_path)
    assert set(inventory) == {"tokenizer_artifact"}
    weights = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(1)}, weights, metadata={"source": "/home/private/weights"})
    with pytest.raises(ValueError, match="private path or credential"):
        audit_public_files(tmp_path)
    weights.unlink()
    (tmp_path / "config.json").write_text(json.dumps({"token": "hf_" + "a" * 32}))
    with pytest.raises(ValueError, match="private path or credential"):
        audit_public_files(tmp_path)
    (tmp_path / "config.json").unlink()
    (tmp_path / "run_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unpublishable file"):
        audit_public_files(tmp_path)
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies/manifest.json").write_text(json.dumps({
        "format_version": 1, "policies": [{"directory": "G05_FAST_LIBERO"}],
    }))
    with pytest.raises(ValueError, match="exactly the six official"):
        PUBLISH.main(["--root", str(tmp_path)])
    import httpx

    request = httpx.Request(
        "POST", "https://huggingface.co/api/models/example/model/commit/main",
        content=iter([b"first\n", b"second\n"]),
    )
    assert request.headers["transfer-encoding"] == "chunked"
    PUBLISH.buffer_commit_metadata(request)
    assert request.content == b"first\nsecond\n"
    assert request.headers["content-length"] == str(len(request.content))
    assert "transfer-encoding" not in request.headers
    mirrored = httpx.Request("POST", "https://hf-mirror.org/api/complete_multipart", content=b"{}")
    PUBLISH.buffer_commit_metadata(mirrored)
    assert mirrored.url.host == "hf-mirror.com" and mirrored.headers["host"] == "hf-mirror.com"
    assert mirrored.content == b"{}"


@pytest.mark.slow
def test_g05_training_generation_and_save_roundtrip(tmp_path, sparse_codec):
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerFast

    from actioncodec2.vla.g05_hf import register_huggingface_auto_classes
    from actioncodec2.vla.g05_hf.configuration_g05 import G05Config
    from actioncodec2.vla.g05_hf.modeling_g05 import G05Model
    from actioncodec2.vla.g05_hf.processing_g05 import G05Processor

    register_huggingface_auto_classes()
    config = G05Config(
        action_vocab_size=8,
        action_token_begin_idx=100,
        text_vocab_size=256,
        image_size=(32, 32),
        max_prompt_length=64,
        max_task_tokens=8,
        max_action_tokens=4,
        text_hidden_size=32,
        text_intermediate_size=64,
        text_num_layers=2,
        text_num_heads=2,
        text_num_kv_heads=1,
        text_head_dim=16,
        text_layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        vision_depth=1,
        vision_hidden_size=16,
        vision_intermediate_size=32,
        vision_num_heads=2,
        vision_patch_size=8,
        vision_spatial_merge_size=2,
        vision_temporal_patch_size=2,
        mrope_section=(2, 2, 2),
        attn_implementation="eager",
        vision_attn_implementation="eager",
        pad_token_id=0,
        eos_token_id=10,
        image_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
        eov_token_id=14,
        state_token_id=15,
    )
    model = G05Model(config)
    model.save_pretrained(tmp_path)
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "task": 2}
    token_backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    token_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=token_backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    G05Processor(tokenizer, config).save_pretrained(tmp_path)

    restored = AutoModel.from_pretrained(tmp_path, local_files_only=True)
    restored_processor = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert type(restored).__name__ == "G05Model"
    assert restored.config.model_type == G05Config.model_type
    assert restored.config.attn_implementation == config.attn_implementation
    saved_config = config.to_dict()
    assert G05Config.from_dict(saved_config, attn_implementation="sdpa").attn_implementation == "sdpa"
    assert saved_config == config.to_dict()
    assert isinstance(restored_processor, G05Processor)
    processor_config = restored_processor.config.to_dict()
    model_config = restored.config.to_dict()
    processor_config.pop("_name_or_path", None)
    model_config.pop("_name_or_path", None)
    assert processor_config == model_config
    assert restored.get_input_embeddings() is restored.get_output_embeddings()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    batch = restored_processor(
        "task",
        torch.zeros(1, 3, 32, 32, dtype=torch.uint8),
        action_labels=torch.tensor([[0, 1]]),
    )
    output = restored(**batch)
    assert torch.isfinite(output.loss)
    assert output.logits.shape[:2] == batch["action_labels"].shape
    model.eval()
    torch.testing.assert_close(model(**batch).logits, output.logits, rtol=0, atol=0)
    output.loss.backward()
    gradients = [p.grad for p in restored.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)

    class FixedGrammar:
        vocab_size = config.action_vocab_size
        maximum_tokens_per_action = 2

        def next_token_mask(self, prefix):
            allowed = torch.zeros(self.vocab_size, dtype=torch.bool)
            allowed[len(prefix)] = True
            return allowed

        def is_complete(self, prefix):
            return len(prefix) == self.maximum_tokens_per_action

    generated = restored.generate_action_tokens(
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        FixedGrammar(),
        action_history_position_mask=batch["action_history_position_mask"],
        proprio_position_mask=batch["proprio_position_mask"],
        do_sample=False,
    )
    assert len(generated) == 1
    assert FixedGrammar().is_complete(generated[0].tolist())
    from actioncodec2.tokenizers.actioncodec2_sparse import ActionCodec2SparseTokenizer

    codec, artifact = sparse_codec
    sparse_adapter = ActionCodec2SparseTokenizer(
        codec.for_action_space("single_eef_delta"), artifact=artifact, time_horizon=4
    )
    sparse_config = config.to_dict() | {
        "action_vocab_size": sparse_adapter.vocab_size, "text_vocab_size": 1024,
        "max_action_tokens": sparse_adapter.max_tokens_per_chunk,
    }
    sparse_config = G05Config.from_dict(sparse_config)
    sparse_model = G05Model(sparse_config).eval()
    boundary = np.zeros((1, 7), np.float32) if sparse_adapter.requires_previous_action else None
    targets = sparse_adapter.encode_batch(np.zeros((1, 4, 7), np.float32), boundary)
    sparse_batch = G05Processor(tokenizer, sparse_config)(
        "task", torch.zeros(1, 3, 32, 32, dtype=torch.uint8),
        action_labels=torch.tensor(targets),
    )
    sparse_output = sparse_model(**sparse_batch)
    assert torch.isfinite(sparse_output.loss)
    sparse_output.loss.backward()
    sparse_generated = sparse_model.generate_action_tokens(
        sparse_batch["input_ids"], sparse_batch["attention_mask"], sparse_batch["pixel_values"],
        sparse_adapter.grammar, do_sample=False,
        action_history_position_mask=sparse_batch["action_history_position_mask"],
        proprio_position_mask=sparse_batch["proprio_position_mask"],
    )
    sparse_rows = [row.tolist() for row in sparse_generated]
    assert all(sparse_adapter.grammar.is_complete(row) for row in sparse_rows)
    assert sparse_adapter.decode_batch(sparse_rows, boundary).shape == (1, 4, 7)
    from actioncodec2.vla.runtime.g05 import load_g05_model
    from safetensors.torch import load_file, save_file

    strict = load_g05_model(tmp_path, device="cpu")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(strict.state_dict()[name], value, rtol=0, atol=0)
    from actioncodec2.vla.g05_hf.modeling_g05 import (
        apply_liger_kernels, liger_kernel_available, qwen_kernels,
    )
    if torch.cuda.is_available() and liger_kernel_available():
        import copy

        classes = (qwen_kernels.Qwen3_5RMSNorm, qwen_kernels.Qwen3_5MLP)
        first = copy.deepcopy(strict).to("cuda")
        apply_liger_kernels(first, device="cuda")
        assert classes == (qwen_kernels.Qwen3_5RMSNorm, qwen_kernels.Qwen3_5MLP)
        second = copy.deepcopy(strict).to("cuda")
        apply_liger_kernels(second, device="cuda")
        for name, value in strict.state_dict().items():
            torch.testing.assert_close(first.state_dict()[name].cpu(), value, rtol=0, atol=0)
        device_batch = {key: value.to("cuda") for key, value in batch.items()}
        with torch.inference_mode():
            first_output = first(**device_batch)
            second_output = second(**device_batch)
        torch.testing.assert_close(first_output.logits, second_output.logits, rtol=0, atol=0)
        assert torch.isfinite(first_output.loss)
        subsequent_cpu = G05Model(config)
        assert torch.isfinite(subsequent_cpu(**batch).loss)
    weights_path = tmp_path / "model.safetensors"
    weights = load_file(weights_path)
    weights.pop(next(key for key in weights if key.endswith("norm.weight")))
    save_file(weights, weights_path, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="weights do not match"):
        load_g05_model(tmp_path, device="cpu")


@pytest.mark.parametrize("benchmark", ["libero", "libero-plus", "simplerenv"])
def test_benchmark_help_shows_only_applicable_options(capsys, benchmark) -> None:
    with pytest.raises(SystemExit) as error:
        CLIENT.main(["--benchmark", benchmark, "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--server" in help_text and "--episodes" in help_text
    if benchmark == "simplerenv":
        assert "--task " in help_text
        assert "--suite" not in help_text and "--task-id" not in help_text
        assert "--libero-config" not in help_text
    else:
        assert "--suite" in help_text and "--task-id" in help_text
        assert "--libero-config" in help_text
        assert "--task " not in help_text


def test_client_executes_stride_and_sends_last_action_as_next_boundary() -> None:
    class Runtime:
        metadata = {
            "requires_previous_action": True,
            "execution_horizon": 2,
            "action_spec": {"dimension": 7},
        }

        def __init__(self) -> None:
            self.requests = []

        def predict(self, params):
            self.requests.append(params)
            actions = np.arange(21, dtype=np.float32).reshape(3, 7)
            return {"actions": pack_array(actions), "tokens": []}

    runtime = Runtime()
    with serve(make_handler(runtime), "127.0.0.1", 0, ping_interval=None) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            port = server.socket.getsockname()[1]
            client = CLIENT.PolicyClient(f"ws://127.0.0.1:{port}")
            try:
                client.reset()
                image = np.zeros((4, 4, 3), dtype=np.uint8)
                actions = [client.next_action([image], "task", None) for _ in range(3)]
                assert len(runtime.requests) == 2
                np.testing.assert_array_equal(actions[0], np.arange(7))
                np.testing.assert_array_equal(actions[1], np.arange(7, 14))
                np.testing.assert_array_equal(runtime.requests[1]["previous_action"], actions[1])
                assert len(runtime.requests[1]["frames"]) == 1
            finally:
                client.close()
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    "low, high, open_is_high, threshold",
    [(0.0, 1.0, False, 0.5), (0.0, 1.0, True, 0.5), (-1.0, 1.0, False, 0.0)],
)
def test_gripper_commands_follow_the_checkpoint_convention(
    low: float, high: float, open_is_high: bool, threshold: float
) -> None:
    spec = {
        "gripper_open": high if open_is_high else low,
        "gripper_closed": low if open_is_high else high,
        "gripper_binary_threshold": threshold,
    }
    opened = np.array([0, 0, 0, 0, 0, 0, spec["gripper_open"]], dtype=np.float32)
    closed = np.array([0, 0, 0, 0, 0, 0, spec["gripper_closed"]], dtype=np.float32)
    assert CLIENT._libero_action(opened, spec)[-1] == -1.0  # LIBERO: -1 opens
    assert CLIENT._libero_action(closed, spec)[-1] == 1.0
    assert CLIENT._simplerenv_action(opened, spec)[-1] == 1.0  # WidowX: +1 opens
    assert CLIENT._simplerenv_action(closed, spec)[-1] == -1.0


def test_libero_state_keeps_the_axis_angle_beyond_pi() -> None:
    """The policy's proprio is robosuite's axis-angle, not SciPy's folded rotation vector."""
    angle = 3.5  # a half-turn plus a bit: a common LIBERO end-effector pose
    quat = np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)])
    axis_angle = CLIENT._quat2axisangle(quat)
    np.testing.assert_allclose(axis_angle, [angle, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(CLIENT._quat2axisangle([0, 0, 0, 1]), 0.0)


def test_libero_plus_robot_initial_state_keeps_scene_state() -> None:
    robot = SimpleNamespace(
        _ref_joint_pos_indexes=[0, 2], robot_model=SimpleNamespace(init_qpos=np.array([2.0, 3.0]))
    )
    env = SimpleNamespace(env=SimpleNamespace(
        robots=[robot], sim=SimpleNamespace(model=SimpleNamespace(nq=4))
    ))
    state = np.arange(10, dtype=np.float32)
    converted = CLIENT._with_robot_qpos(env, state)
    changed = np.array(robot._ref_joint_pos_indexes) + 1
    np.testing.assert_array_equal(converted[changed], robot.robot_model.init_qpos)
    untouched = np.ones(len(state), dtype=bool)
    untouched[changed] = False
    np.testing.assert_array_equal(converted[untouched], state[untouched])
    np.testing.assert_array_equal(state, np.arange(len(state), dtype=np.float32))


def test_concurrent_clients_are_batched_on_one_inference_thread():
    """Torch builds a thread pool per calling thread, and a batch costs about one row."""
    from actioncodec2.vla.runtime.server import PolicyRuntime

    threads, batch_sizes = set(), []

    class Policy:
        def predict_batch(self, observations, *, temperature):
            threads.add(threading.get_ident())
            batch_sizes.append(len(observations))
            return [(np.full((1, 1), float(row["task"]), np.float32), [int(row["task"])])
                    for row in observations]

    runtime = type("FakeRuntime", (), {})()
    runtime.request_timeout = 30.0
    runtime.batcher = MicroBatcher(Policy(), device="cpu", max_batch_size=8, wait_seconds=0.2)
    runtime.observation = PolicyRuntime.observation
    runtime.predict = PolicyRuntime.predict.__get__(runtime)
    frame = pack_array(np.zeros((4, 4, 3), dtype=np.uint8))
    clients = 6
    ready = threading.Barrier(clients)
    with serve(make_handler(runtime), "127.0.0.1", 0, ping_interval=None) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            port = server.socket.getsockname()[1]

            def client(index: int) -> None:
                with connect(f"ws://127.0.0.1:{port}", proxy=None) as socket:
                    ready.wait()
                    params = {"frames": [frame], "task": str(index)}
                    socket.send(json.dumps({"id": index, "method": "predict", "params": params}))
                    result = json.loads(socket.recv())["result"]
                    assert result["tokens"] == [index]  # each client gets its own answer

            with ThreadPoolExecutor(max_workers=clients) as pool:
                for future in [pool.submit(client, index) for index in range(clients)]:
                    future.result()
            with connect(f"ws://127.0.0.1:{port}", proxy=None) as socket:
                socket.send(json.dumps({"id": "error", "method": "/private/internal-method"}))
                response = json.loads(socket.recv())
                assert response == {"id": "error", "error": "request failed; see server logs"}
            with pytest.raises(ValueError, match="finite"):
                runtime.batcher.submit({}, float("nan"))
        finally:
            server.shutdown()
            thread.join()
    assert len(threads) == 1
    assert sum(batch_sizes) == clients and max(batch_sizes) > 1


def _artifact(order):
    configured = os.environ.get("ACTIONCODEC2_TEST_ARTIFACT_ROOT")
    if not configured:
        pytest.skip("external ActionCodec2 artifact is not installed")
    path = Path(configured) / order
    if not (path / "processor_config.json").is_file():
        pytest.skip("external ActionCodec2 artifact is not installed")
    return path


@pytest.mark.slow
@pytest.mark.parametrize("order", ["first_order", "second_order"])
def test_portable_native_artifact_matches_python_and_survives_save(order, monkeypatch,
                                                                  tmp_path):
    source = tmp_path / "source"
    shutil.copytree(_artifact(order), source)
    local_files = ("AGENTS.md", ".pi/sessions/local.jsonl", "runtime/CLAUDE.md",
                   "runtime/.env", "profiles/eef/.codex/settings.json")
    for name in local_files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local-only")
    from transformers import AutoProcessor
    from actioncodec2.tokenizers.actioncodec2 import _native_kernels

    if _native_kernels() is None:
        pytest.skip("native kernels are not installed")
    output = tmp_path / "portable"
    TOKENIZER_EXPORT.export_bundle(source, output)
    assert all(not (output / name).exists() for name in local_files)
    codec = AutoProcessor.from_pretrained(output, trust_remote_code=True, local_files_only=True)
    rng = np.random.default_rng(42)
    for action_space, width in (("single_eef_delta", 7), ("single_joint7_delta", 8),
                                ("single_joint6_delta", 7),
                                ("dual_eef_delta", 14), ("single_eef_abs", 7)):
        selected = codec.for_action_space(action_space)
        actions = rng.normal(0, 0.002, (4, 20, width)).astype(np.float32)
        for component in selected.action_space.components:
            if component.gripper_dim is not None:
                actions[..., component.gripper_dim] = rng.integers(0, 2, (4, 20))
        kwargs = {}
        if selected.requires_previous_first_order:
            kwargs["previous_first_order"] = selected.first_order_template(4)
        if selected.action_space.input == "absolute":
            kwargs["current_state"] = np.zeros((4, width), dtype=np.float32)
        for rate in (15, 20):
            monkeypatch.setenv("ACTIONCODEC2_NO_NATIVE", "1")
            expected_tokens = selected.encode(actions, fps=rate, **kwargs)
            expected_actions = selected.decode(expected_tokens, fps=rate, **kwargs).numpy()
            grammar = selected.grammar(20, fps=rate)
            mask = grammar.next_token_mask([])
            expected_mask = mask.copy()
            mask[:] = ~mask
            np.testing.assert_array_equal(grammar.next_token_mask([]), expected_mask)
            monkeypatch.delenv("ACTIONCODEC2_NO_NATIVE")
            tokens = selected.encode(actions, fps=rate, **kwargs)
            assert tokens == expected_tokens
            np.testing.assert_array_equal(selected.decode(tokens, fps=rate, **kwargs).numpy(),
                                          expected_actions)
            assert all(selected.grammar(20, fps=rate).is_complete(row) for row in tokens)
            mask = grammar.next_token_mask([])
            np.testing.assert_array_equal(mask, expected_mask)
            mask[:] = ~mask
            np.testing.assert_array_equal(grammar.next_token_mask([]), expected_mask)
            if action_space == "single_joint6_delta":
                dims = selected.action_space.components[0].active_dims
                serializer = selected.action_tokenizer.profiles["joint"].processor.serializer
                full = sum(1 << dim for dim in dims)
                for token in np.flatnonzero(expected_mask):
                    assert all(int(serializer.spatial.support[spatial]) & ~full == 0
                               for _, spatial in serializer.patterns[int(token)])
    saved = tmp_path / "saved"
    codec.save_pretrained(saved)
    assert (saved / "runtime/backends/actioncodec2/physical_codec.py").is_file()
    assert (saved / "actioncodec2_kernels/parallel.hpp").is_file()
    for name in ("inference.cpp", "grammar.cpp", "quaternion.hpp"):
        assert (saved / "actioncodec2_kernels" / name).is_file()
    assert (saved / "setup.py").is_file()
    restored = AutoProcessor.from_pretrained(saved, trust_remote_code=True, local_files_only=True)
    actions = np.zeros((2, 12, 7), np.float32)
    kwargs = {"previous_first_order": codec.first_order_template(2)} if codec.requires_previous_first_order else {}
    assert restored.encode(actions, fps=15, **kwargs) == codec.encode(actions, fps=15, **kwargs)


@pytest.mark.slow
@pytest.mark.parametrize("order", ["first_order", "second_order"])
def test_native_reconstruction_preserves_rotation_conventions(order, monkeypatch):
    from dataclasses import replace
    from itertools import product

    from actioncodec2.tokenizers.actioncodec2 import _load_processor, _native_kernels

    native = _native_kernels()
    if native is None:
        pytest.skip("native kernels are disabled")
    assert hasattr(native, "so3_reconstruct") and hasattr(native, "additive_reconstruct")
    router = _load_processor(_artifact(order))
    profile = router.profiles["eef"].processor
    rng = np.random.default_rng(31)
    for component, original in profile.physical._runtimes:
        if component.primitive_order == 0:
            continue
        settings = (product((0, 1), ("left", "right"), ("rotation_vector", "euler_xyz"))
                    if component.space == "so3" else ((i, None, None) for i in (0, 1, 2)))
        if component.space != "so3" and component.primitive_order != 2:
            continue
        for input_order, multiplication, representation in settings:
            changes = {"input_order": input_order, "input_transform": replace(
                component.input_transform, scale=(0.5, 1.0, 2.0))}
            if component.space == "so3":
                changes.update(multiplication=multiplication, representation=representation)
            runtime = type(original)(replace(component, **changes), original.primitives)
            cells = np.stack([rng.integers(0, len(axis), (7, 23))
                              for axis in runtime.primitives], axis=-1)
            boundary = (rng.normal(0, 0.015, (7, 3)).astype(np.float32)
                        if component.primitive_order == 2 else None)
            monkeypatch.setenv("ACTIONCODEC2_NO_NATIVE", "1")
            expected = runtime.decode_batch(cells, boundary)
            monkeypatch.delenv("ACTIONCODEC2_NO_NATIVE")
            np.testing.assert_array_equal(runtime.decode_batch(cells, boundary), expected)


def test_native_reconstruction_and_grammar_reject_invalid_boundaries():
    from actioncodec2.tokenizers.actioncodec2 import _native_kernels

    native = _native_kernels()
    if native is None:
        pytest.skip("native kernels are disabled")
    table = np.tile([-0.01, 0.0, 0.01], (3, 1)).astype(np.float64)
    bins = np.full(3, 3, np.int32)
    cells = np.ones((2, 4, 3), np.int64)
    previous, scale = np.zeros((2, 3), np.float32), np.ones(3, np.float32)
    valid = [table, bins, cells, previous, scale]
    for index, bad in ((0, table[:, :0]), (1, np.array([3, 2, 3], np.int32)),
                       (2, np.full_like(cells, -1)), (2, np.full_like(cells, 3)),
                       (3, np.full_like(previous, np.nan)), (4, scale * 0),
                       (4, scale * np.inf)):
        args = valid.copy()
        args[index] = bad
        if index == 1:
            args[2] = np.full_like(cells, 2)
        with pytest.raises(ValueError):
            native.so3_reconstruct(*args, 0, 2, False, True, 0.1)
        with pytest.raises(ValueError):
            native.additive_reconstruct(*args, 0)
    for input_order, primitive_order, radius in ((2, 1, 0.1), (0, 0, 0.1),
                                                 (0, 2, np.nan), (0, 2, -1)):
        with pytest.raises(ValueError):
            native.so3_reconstruct(*valid, input_order, primitive_order, False, True, radius)

    grammar = native.SetGrammar(2, 1, [0, 1], [[(0, 1)], [(0, 2)]])
    np.testing.assert_array_equal(grammar.mask((0, 0)), [True, False])
    assert grammar.advance((0, 0), 1) is None
    for state in ((), (0,), (-1, 0), (2, 0)):
        with pytest.raises(ValueError):
            grammar.mask(state)
        with pytest.raises(ValueError):
            grammar.advance(state, 0)
    with pytest.raises(ValueError):
        native.SetGrammar(1, 1, [0], [[(-1, 1)]])
    with pytest.raises(ValueError):
        native.SetGrammar(1, 1, [0], [[(0, 1), (0, 1)]])
    plain = native.PlainGrammar(2, [1, 2], np.ones((1, 2), dtype=bool))
    assert plain.advance((0,), 1) == (2,)
    assert not plain.mask((2,)).any()
    for state in ((), (-1,), (3,), (0, 0)):
        with pytest.raises(ValueError):
            plain.mask(state)


def test_native_reconstruction_and_grammar_work_after_fork():
    import multiprocessing

    from actioncodec2.tokenizers.actioncodec2 import _native_kernels

    native = _native_kernels()
    if native is None or "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires native kernels and fork")
    table = np.tile([-0.01, 0.0, 0.01], (3, 1)).astype(np.float64)
    bins = np.full(3, 3, np.int32)
    cells = np.ones((128, 128, 3), np.int64)
    previous, scale = np.zeros((128, 3), np.float32), np.ones(3, np.float32)
    grammar = native.SetGrammar(128, 7, [0], [[(0, 7)]])
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)

    def run():
        output = native.so3_reconstruct(table, bins, cells, previous, scale, 0, 2,
                                        False, True, 0.1)
        additive = native.additive_reconstruct(table, bins, cells, previous, scale, 0)
        return bool(np.isfinite(output).all() and np.isfinite(additive).all()
                    and grammar.mask((0,) * 128).all())

    assert run()
    child = context.Process(target=lambda: writer.send(run()))
    child.start()
    try:
        child.join(10)
        assert not child.is_alive(), "native kernels hung after fork"
        assert child.exitcode == 0 and reader.poll(1) and reader.recv()
    finally:
        if child.is_alive():
            child.kill()
            child.join()
        reader.close()
        writer.close()


@pytest.fixture(scope="module", params=[1, 2])
def sparse_codec(request, tmp_path_factory):
    from actioncodec2 import ActionCodec2Sparse
    from actioncodec2.core.backends.actioncodec2 import ActionCodec2Config

    fresh = ActionCodec2Sparse(action_space="single_eef_delta", primitive_order=request.param)
    configs = {}
    for name, runtime in fresh.action_tokenizer.profiles.items():
        config = runtime.processor.config.to_dict()
        config["train"]["bpe"]["vocab_size"] = sum(runtime.processor.layout.sparse_bins) + 24
        configs[name] = ActionCodec2Config.from_dict(config)
    codec = ActionCodec2Sparse(action_space="single_eef_delta", profile_configs=configs)
    rng = np.random.default_rng(42)
    for space, width in (("single_eef_delta", 7), ("single_joint7_delta", 8)):
        codec = codec.for_action_space(space)
        actions = rng.normal(0, 0.004, (6, 24, width)).astype(np.float32)
        actions[:, ::3, :-1] = 0
        actions[..., -1] = np.arange(24) % 7 > 3
        codec.fit(actions, fps=15, bootstrap_context=request.param == 2, threads=2)
    directory = tmp_path_factory.mktemp(f"sparse_order{request.param}")
    codec.save_pretrained(directory)
    return codec, directory


def test_sparse_frontend_routing_boundaries_native_and_offline(sparse_codec, monkeypatch, tmp_path):
    from actioncodec2 import ActionCodec2Sparse
    from transformers import AutoProcessor

    fitted, _ = sparse_codec
    for layout in ("single_eef_delta", "dual_eef_delta", "single_joint6_delta", "dual_joint7_delta"):
        codec = fitted.for_action_space(layout)
        rng = np.random.default_rng(81)
        actions = rng.normal(0, 0.003, (2, 12, codec.action_dim)).astype(np.float32)
        for component in codec.action_space.components:
            actions[..., component.gripper_dim] = np.arange(12) > 6
        context = {"fps": 15}
        if codec.requires_previous_first_order:
            with pytest.raises(ValueError, match="previous_first_order"):
                codec.encode(actions, **context)
            context["previous_first_order"] = codec.first_order_template(2)
        rows = codec.encode(actions, **context)
        decoded = codec.decode(rows, horizon=12, **context).numpy()
        assert decoded.shape == actions.shape and np.isfinite(decoded).all()
        assert [row.count(codec.action_tokenizer.end_token) for row in rows] == [int(layout.startswith("dual"))] * 2
        grammar = codec.grammar(12, fps=15)
        for row in rows:
            assert grammar.is_complete(row)
            for index, token in enumerate(row):
                assert grammar.next_token_mask(row[:index])[token]
        with monkeypatch.context() as patch:
            patch.setenv("ACTIONCODEC2_NO_NATIVE", "1")
            assert codec.encode(actions, **context) == rows
            np.testing.assert_array_equal(codec.decode(rows, horizon=12, **context), decoded)
        for invalid in ([True], [0.5], [-1], [codec.vocab_size], [2**70]):
            with pytest.raises((ValueError, OverflowError)):
                codec.decode([invalid], horizon=12, **{key: value if key == "fps" else {k: v[:1] for k, v in value.items()} for key, value in context.items()})
        for cut in range(1, len(rows[0]) + 1):
            steps = grammar.covered_steps(rows[0][:cut])
            if steps:
                prefix_context = dict(context)
                if "previous_first_order" in context:
                    prefix_context["previous_first_order"] = {k: v[:1] for k, v in context["previous_first_order"].items()}
                prefix = codec.decode_prefix([rows[0][:cut]], horizon=12, steps=steps, **prefix_context)
                np.testing.assert_array_equal(prefix[0], decoded[0, :steps])
        codec.save_pretrained(tmp_path)
        assert not (tmp_path / "fit_report.json").exists()
        for loader in (ActionCodec2Sparse, AutoProcessor):
            reloaded = loader.from_pretrained(tmp_path, local_files_only=True)
            assert reloaded.encode(actions, **context) == rows
            np.testing.assert_array_equal(reloaded.decode(rows, horizon=12, **context), decoded)
    # Absolute inputs must carry their observed source state.
    absolute = fitted.for_action_space("single_eef_abs")
    state = np.zeros((1, 7), np.float32)
    actions = np.zeros((1, 8, 7), np.float32)
    kwargs = {"fps": 15}
    if absolute.requires_previous_first_order:
        kwargs["previous_first_order"] = absolute.first_order_template(1)
    with pytest.raises(ValueError, match="current_state"):
        absolute.encode(actions, **kwargs)
    rows = absolute.encode(actions, current_state=state, **kwargs)
    np.testing.assert_array_equal(absolute.decode(rows, horizon=8, current_state=state, **kwargs), actions)


def test_sparse_static_velocity_and_explicit_stop(sparse_codec):
    import torch
    from actioncodec2 import build_tokenizer
    from actioncodec2.tokenizers.actioncodec2_sparse import ActionCodec2SparseTokenizer

    fitted, directory = sparse_codec
    codec = fitted.for_action_space("single_eef_delta")
    actions = np.zeros((1, 10, 7), np.float32)
    kwargs = {"fps": 15}
    previous = None
    if codec.requires_previous_first_order:
        actions[..., 0] = 0.01
        previous = actions[:, 0].copy()
        kwargs["previous_first_order"] = {"arm": previous}
    rows = codec.encode(actions, **kwargs)
    assert len(rows[0]) == 1
    np.testing.assert_allclose(codec.decode(rows, horizon=10, **kwargs), actions, atol=1e-7)
    adapter = ActionCodec2SparseTokenizer(codec, artifact=directory, time_horizon=10)
    streams = adapter.encode_batch(actions, previous)
    assert adapter.grammar.inner.is_complete([t + adapter.grammar.offset for t in streams[0][:-1]])
    assert not adapter.grammar.is_complete(streams[0][:-1])
    assert adapter.grammar.next_token_mask(streams[0][:-1])[-1]
    assert adapter.grammar.is_complete(streams[0])
    assert not adapter.grammar.next_token_mask(streams[0]).any()
    np.testing.assert_allclose(adapter.decode_batch(streams, previous), actions, atol=1e-7)
    with pytest.raises(ValueError, match="EOS"):
        adapter.decode_batch([streams[0][:-1]], previous)
    restored = build_tokenizer(adapter.to_json())
    assert restored.encode_batch(actions, previous) == streams
    bad = adapter.to_json() | {"artifact_fingerprint": "0" * 64}
    with pytest.raises(ValueError, match="fingerprint"):
        build_tokenizer(bad)
    processor = codec.logits_processor(10, fps=15, prompt_length=1, eos_token_id=codec.vocab_size)
    scores = torch.zeros(1, codec.vocab_size + 1)
    assert not torch.isfinite(processor(torch.tensor([[0]]), scores)[0, -1])
    assert torch.isfinite(processor(torch.tensor([[0, *rows[0]]]), scores)[0, -1])


@pytest.mark.parametrize("spatial_merges", [0, 6])
def test_sparse_native_training_and_cell_serialization_match_reference(spatial_merges, monkeypatch):
    from actioncodec2.core.backends.actioncodec2.sparse import SparseLayout, fit_sparse_vocabulary

    layout = SparseLayout((5, 5, 2), (2, 2, None))
    rng = np.random.default_rng(5)
    chunks = []
    for _ in range(24):
        cells = np.column_stack((rng.integers(0, 5, (12, 2)), np.repeat(rng.integers(0, 2), 12)))
        cells[::2, :2] = 2
        chunks.append(cells)
    atoms = sum(layout.sparse_bins)
    options = dict(spatial_vocab_size=atoms + spatial_merges, vocab_size=atoms + spatial_merges + 20, lookahead=4)
    reference, _ = fit_sparse_vocabulary(layout, chunks, backend="python", **options)
    native, _ = fit_sparse_vocabulary(layout, chunks, backend="native", threads=2, **options)
    assert native.to_dict() == reference.to_dict()
    batch = np.stack(chunks)
    rows = native.encode_batch(batch, threads=2)
    np.testing.assert_array_equal(native.decode_batch(rows, 12, (True,) * 3), batch)
    with monkeypatch.context() as patch:
        patch.setenv("ACTIONCODEC2_NO_NATIVE", "1")
        assert native.encode_batch(batch) == rows
        np.testing.assert_array_equal(native.decode_batch(rows, 12, (True,) * 3), batch)


def test_sparse_rejects_corrupt_rules_and_cells_before_native_casts():
    from actioncodec2.core.backends.actioncodec2.sparse import SparseLayout, SparseVocabulary, fit_sparse_vocabulary

    layout = SparseLayout((3, 2), (1, None))
    vocabulary = SparseVocabulary(layout, (), (), 4)
    for row in ([True], [0.25], ["0"], [-1], [2**64]):
        with pytest.raises(ValueError):
            vocabulary.decode_batch([row], 4, (True, True))
    for spatial, temporal in ((((-1, 2),), ()), ((), ((-1, 0, 0),)), ((), ((0.1, 1, 0),))):
        with pytest.raises(ValueError):
            SparseVocabulary(layout, spatial, temporal, 4)
    cells = np.ones((1, 4, 2), dtype=np.int64)
    cells[0, 0, 0] = 2**32 + 1
    with pytest.raises(ValueError, match="primitive range"):
        vocabulary.encode_batch(cells)
    with pytest.raises(ValueError, match="primitive range"):
        fit_sparse_vocabulary(layout, [cells[0]], spatial_vocab_size=5, vocab_size=8, lookahead=4)


@pytest.mark.skipif(sys.platform != "linux", reason="requires fork")
def test_sparse_native_inference_survives_fork_after_fitting(sparse_codec):
    import multiprocessing

    fitted, _ = sparse_codec
    codec = fitted.for_action_space("single_eef_delta")
    values = np.zeros((16, 8, 7), np.float32)
    context = {"fps": 15}
    if codec.requires_previous_first_order:
        context["previous_first_order"] = codec.first_order_template(len(values))
    expected = codec.encode(values, **context)
    receive, send = multiprocessing.get_context("fork").Pipe(duplex=False)

    def worker():
        rows = codec.encode(values, **context)
        send.send(bool(rows == expected and np.isfinite(codec.decode(rows, horizon=8, **context).numpy()).all()))
        send.close()

    process = multiprocessing.get_context("fork").Process(target=worker)
    process.start()
    send.close()
    try:
        process.join(timeout=15)
        assert not process.is_alive(), "native Sparse inference stalled after fork"
        assert process.exitcode == 0 and receive.poll() and receive.recv()
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        receive.close()


def test_sparse_policy_export_is_self_contained(sparse_codec, tmp_path, monkeypatch):
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models
    from actioncodec2.tokenizers.actioncodec2_sparse import ActionCodec2SparseTokenizer
    from actioncodec2.vla.runtime.run import load_runtime_config, build_tokenizer_for

    codec, artifact = sparse_codec
    adapter = ActionCodec2SparseTokenizer(codec.for_action_space("single_eef_delta"), artifact=artifact, time_horizon=8)
    source = tmp_path / "checkpoint"
    source.mkdir()
    Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(str(source / "tokenizer.json"))
    save_file({"test_tensor": np.zeros((1,), np.float32)}, source / "model.safetensors")
    from actioncodec2.vla.g05_hf.constants import G05_MODEL_TYPE
    (source / "config.json").write_text(json.dumps({
        "model_type": G05_MODEL_TYPE,
        "continuous_action_dim": 7 if adapter.requires_previous_action else 0,
    }))
    (source / "tokenizer_config.json").write_text('{}')
    (source / "policy_config.json").write_text(json.dumps({
        "action_vocab_size": adapter.vocab_size, "action_token_begin": 8, "total_vocab_size": 4096,
        "execution_stride": 1, "requires_previous_action": adapter.requires_previous_action,
    }))
    config = RunConfig(
        ModelConfig(path="./base", tokenizer_path="./text", proprio_mode="none"),
        adapter.to_json(), DataConfig(roots=[], horizon=8, control_frequency_hz=15), TrainConfig(output_dir="./train"),
    )
    config.save(source / "run_config.json")
    bundle = POLICY_EXPORT.export_checkpoint(source, tmp_path / "bundle")
    assert not list(bundle.rglob("fit_report.json"))
    assert not list(bundle.rglob("run_config.json"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reloaded = build_tokenizer_for(load_runtime_config(bundle))
    assert Path(reloaded.artifact).is_relative_to(bundle)
    x = np.zeros((1, 8, 7), np.float32)
    boundary = np.zeros((1, 7), np.float32) if adapter.requires_previous_action else None
    rows = adapter.encode_batch(x, boundary)
    assert reloaded.encode_batch(x, boundary) == rows
    np.testing.assert_array_equal(reloaded.decode_batch(rows, boundary), adapter.decode_batch(rows, boundary))
    POLICY_EXPORT.verify_bundle(bundle)
    assert "actioncodec2-0.1.0.tar.gz" in (bundle / "README.md").read_text()


@pytest.mark.parametrize("sparse", [False, True])
def test_shared_slots_and_opt_in_pretraining_cleaning(sparse, tmp_path):
    from actioncodec2 import ActionCodec2, ActionCodec2Sparse, PreprocessingSpec
    from actioncodec2.core.frontend.fitting import ProfileFitter
    from actioncodec2.core.frontend.training import TrainingMixin

    cls = ActionCodec2Sparse if sparse else ActionCodec2
    codec = cls(action_space="single_eef_delta")
    assert cls.preprocess is TrainingMixin.preprocess
    assert cls.fit is TrainingMixin.fit
    assert len(codec.action_spaces) == 13
    for preset in codec.action_spaces.values():
        columns = [d for c in preset.components for d in c.all_source_dims]
        assert sorted(columns) == list(range(preset.input_dim))
        assert preset.token_order == "time_major"
        for component in preset.components:
            assert component.active_dims[-1] == (6 if component.space == "eef" else 7) if component.gripper_dim is not None else True
            if component.space == "joint" and len(component.joint_dims) == 6 and component.gripper_dim is not None:
                assert component.active_dims == (0, 1, 2, 3, 4, 5, 7)
    alias = codec.for_action_space("single_joint6_abs_continuous_gripper")
    assert alias.action_space.name == "single_joint6_abs_no_gripper"
    assert alias.action_space.components[0].gripper_dim is None

    actions = np.zeros((1, 20, 7), np.float32)
    actions[..., 0] = 0.004
    actions[:, 10, 0] = 0.014
    actions[..., 3] = 0.002
    actions[:, 8:, 6] = 1
    record = {"actions": actions, "fps": 15}
    original = ProfileFitter(codec)._route(record)["eef"][0].values.copy()
    spec = PreprocessingSpec(profile="eef", trim_mode="off", savgol_window=9)
    codec.preprocessing = {"eef": spec}
    np.testing.assert_array_equal(ProfileFitter(codec)._route(record)["eef"][0].values, original)
    cleaned = ProfileFitter(codec, apply_preprocessing=True)._route(record)["eef"][0].values
    assert not np.array_equal(cleaned[:, :3], original[:, :3])
    np.testing.assert_array_equal(cleaned[:, 3:], original[:, 3:])
    np.testing.assert_array_equal(cleaned[0], original[0])
    np.testing.assert_array_equal(cleaned, codec.preprocess(original, gripper_column=-1))
    with pytest.raises(ValueError, match="requires apply_preprocessing"):
        ProfileFitter(codec, preprocessing={"eef": spec})
    codec.fit(actions, fps=15, apply_preprocessing=True, threads=2)
    assert codec.training_report["apply_preprocessing"] is True
    rows = codec.encode(actions, fps=15)
    codec.preprocessing = {}
    assert codec.encode(actions, fps=15) == rows
    codec.preprocessing = {"eef": spec}
    codec.save_pretrained(tmp_path)
    restored = cls.from_pretrained(tmp_path, local_files_only=True)
    assert restored.encode(actions, fps=15) == rows
    assert restored.preprocessing["eef"] == spec
    assert not (tmp_path / "fit_report.json").exists()
