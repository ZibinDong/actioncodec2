"""Load a named ActionCodec2 variant or a standalone saved processor."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys


PROFILE_FILES = ("config.yaml", "primitives.npz", "bpe.json", "config.json", "processor_config.json")
SHARED_PATTERNS = (
    "config.json", "processor_config.json", "processing_actioncodec2.py", "variants.json",
    "runtime/**", "actioncodec2_kernels/**", "setup.py", "pyproject.toml", "LICENSE",
    "README.md", "requirements.txt",
)

# Published identities: aliases never change token IDs or asset fingerprints.
VARIANT_ALIASES = {
    **{f"paper-{order}": f"paper-o{index}-5mm-4k-2bin-1p2ms"
       for index, order in enumerate(("1st", "2nd"), 1)},
    **{f"extended-{order}-{precision}-{size}k":
       f"extended-o{index}-{precision}-{size}k-2bin-1p2ms"
       for index, order in enumerate(("1st", "2nd"), 1)
       for precision in ("1mm", "5mm", "10mm") for size in (1, 2, 4, 8, 16)
       if precision != "1mm" or size != 1},
    **{f"sparse-{dynamics}-{mode}": f"sparse-o1-5mm-4k-{mode}-{speed}"
       for dynamics, speed in (("standard", "1p2ms"), ("high-dynamic", "2ms"))
       for mode in ("2bin", "21bin")},
}


def resolve_variant_name(variant, variants):
    """Resolve either published spelling against an old or new catalog."""
    if variant in variants:
        return variant
    canonical = VARIANT_ALIASES.get(variant, variant)
    if canonical in variants:
        return canonical
    for legacy, name in VARIANT_ALIASES.items():
        if name == canonical and legacy in variants:
            return legacy
    raise ValueError(f"select variant= from: {', '.join(sorted(variants))}")


def asset_inventory(directory: Path) -> dict[str, str]:
    """Hash the complete configuration and fitted assets for one variant."""
    sparse = (directory / "sparse_router_config.yaml").is_file()
    names = ["sparse_router_config.yaml" if sparse else "router_config.yaml", "config.json", "processor_config.json"]
    profiles = [name for name in ("eef", "joint") if (directory / "profiles" / name).is_dir()]
    if not profiles:
        raise ValueError("variant has no fitted profiles")
    names.extend(f"profiles/{profile}/{name}" for profile in profiles
                 for name in (tuple("sparse_bpe.json" if n == "bpe.json" else n for n in PROFILE_FILES) if sparse else PROFILE_FILES))
    names.extend(f"profiles/{profile}/preprocessing.json" for profile in profiles
                 if (directory / "profiles" / profile / "preprocessing.json").is_file())
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in sorted(names)}


def inventory_fingerprint(files: dict[str, str]) -> str:
    """Return the stable identity of a fitted asset inventory."""
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_inventory(directory: Path, record: dict) -> None:
    files = asset_inventory(directory)
    if files != record.get("files") or inventory_fingerprint(files) != record.get("fingerprint"):
        raise ValueError("ActionCodec2 asset fingerprint mismatch; restore the complete selected variant")


class _VariantBinding:
    @property
    def variant(self):
        return self._variant_origin.get("name")

    @property
    def variant_fingerprint(self):
        return self._variant_origin.get("fingerprint")

    def for_action_space(self, name):
        codec = super().for_action_space(name)
        codec._variant_origin = dict(self._variant_origin)
        return codec

    def fit(self, *args, **kwargs):
        result = super().fit(*args, **kwargs)
        self._variant_origin = {}
        return result

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        directory = Path(save_directory)
        include_training_report = kwargs.pop("include_training_report", False)
        if (directory / "variants.json").is_file():
            raise ValueError("save a selected variant to a separate directory, outside the catalog root")
        if not push_to_hub and kwargs:
            raise TypeError(f"unused save_pretrained keyword(s): {', '.join(sorted(kwargs))}")
        files = super().save_pretrained(directory, include_training_report=include_training_report)
        inventory = asset_inventory(directory)
        manifest = directory / "artifact_manifest.json"
        manifest.write_text(json.dumps({
            "format_version": 1,
            "origin": self._variant_origin,
            "files": inventory,
            "fingerprint": inventory_fingerprint(inventory),
        }, indent=2) + "\n", encoding="utf-8")
        files.append(str(manifest))
        if push_to_hub:
            package = type(self).__module__.split(".frontend.", 1)[0]
            importlib.import_module(f"{package}.backends.actioncodec2.hf").upload_pretrained_directory(directory, **kwargs)
        return files


class ActionCodec2:
    """Hugging Face entry point for catalog and standalone artifacts."""

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        return cls

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, variant=None,
                        variant_fingerprint=None, action_space=None, **kwargs):
        source = Path(pretrained_model_name_or_path).expanduser()
        kwargs.pop("_from_auto", None)
        kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        if subfolder:
            raise ValueError("select an ActionCodec2 catalog variant with variant=, not subfolder=")
        allowed = {"cache_dir", "force_download", "local_files_only", "token", "revision", "repo_type"}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"unsupported from_pretrained keyword(s): {', '.join(unknown)}")
        if variant is not None and (not isinstance(variant, str) or
                                    re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) is None):
            raise ValueError("variant must be a published variant name")
        if variant_fingerprint is not None and (
            not isinstance(variant_fingerprint, str) or
            re.fullmatch(r"[0-9a-f]{64}", variant_fingerprint) is None
        ):
            raise ValueError("variant_fingerprint must be a SHA-256 fingerprint")
        remote = not source.is_dir()
        if remote and str(pretrained_model_name_or_path).startswith(("/", "./", "../", "~")):
            raise FileNotFoundError(f"ActionCodec2 artifact does not exist: {source}")
        revision = None
        if remote:
            from huggingface_hub import hf_hub_download, snapshot_download

            code_revision = Path(__file__).parent.name
            if re.fullmatch(r"[0-9a-f]{40}", code_revision):
                requested = kwargs.get("revision")
                if requested and re.fullmatch(r"[0-9a-f]{40}", requested) and requested != code_revision:
                    raise ValueError("ActionCodec2 code and assets must use the same revision")
                kwargs["revision"] = code_revision
            metadata_file = Path(hf_hub_download(
                str(pretrained_model_name_or_path), "processor_config.json", **kwargs
            ))
            root = metadata_file.parent
            revision = root.name
            kwargs["revision"] = revision
        else:
            root = source
            metadata_file = root / "processor_config.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("invalid ActionCodec2 processor configuration")
        record = None
        origin = {}
        if metadata.get("variant_catalog") == "variants.json":
            catalog_path = (Path(hf_hub_download(
                str(pretrained_model_name_or_path), "variants.json", **kwargs
            )) if remote else root / "variants.json")
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            if (not isinstance(catalog, dict) or catalog.get("format_version") != 1 or
                    not isinstance(catalog.get("variants"), dict)):
                raise ValueError("unsupported ActionCodec2 variant catalog")
            variants = catalog["variants"]
            variant = resolve_variant_name(variant, variants)
            record = variants[variant]
            if not isinstance(record, dict):
                raise ValueError("invalid ActionCodec2 variant record")
            directory = root / "variants" / variant
            if remote:
                root = Path(snapshot_download(
                    str(pretrained_model_name_or_path),
                    allow_patterns=[*SHARED_PATTERNS, f"variants/{variant}/**"], **kwargs,
                ))
                directory = root / "variants" / variant
            _check_inventory(directory, record)
            origin = {"name": variant, "fingerprint": record["fingerprint"]}
            if remote:
                origin.update(repo_id=str(pretrained_model_name_or_path), revision=revision)
        else:
            if remote:
                root = Path(snapshot_download(
                    str(pretrained_model_name_or_path),
                    allow_patterns=[*SHARED_PATTERNS, "router_config.yaml", "profiles/**",
                                    "sparse_router_config.yaml", "artifact_manifest.json"], **kwargs,
                ))
            directory = root
            manifest = root / "artifact_manifest.json"
            if manifest.is_file():
                record = json.loads(manifest.read_text(encoding="utf-8"))
                if (not isinstance(record, dict) or record.get("format_version") != 1 or
                        not isinstance(record.get("origin"), dict)):
                    raise ValueError("unsupported ActionCodec2 saved artifact manifest")
                _check_inventory(directory, record)
                origin = record["origin"]
            if variant is not None and VARIANT_ALIASES.get(variant, variant) != VARIANT_ALIASES.get(origin.get("name"), origin.get("name")):
                raise ValueError("saved processor contains a different variant; load the catalog to switch")
        if variant_fingerprint is not None and variant_fingerprint != origin.get("fingerprint"):
            raise ValueError("variant fingerprint differs from the tokenizer bound to this policy")

        family = "actioncodec2_sparse" if (directory / "sparse_router_config.yaml").is_file() else "actioncodec2"
        if __package__ == "actioncodec2.core":
            package_name = __package__
        else:
            runtime = root / "runtime"
            package_file = runtime / "__init__.py"
            if not package_file.is_file():
                raise FileNotFoundError(f"ActionCodec2 runtime is missing from {root}")
            identity = hashlib.sha256(str(runtime.resolve()).encode()).hexdigest()[:16]
            package_name = f"_actioncodec2_artifact_{identity}"
            if package_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    package_name, package_file, submodule_search_locations=[str(runtime)]
                )
                package = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = package
                spec.loader.exec_module(package)
        processor = importlib.import_module(f"{package_name}.frontend.{family}.processor")
        base = processor.ActionCodec2Sparse if family == "actioncodec2_sparse" else processor.ActionCodec2
        if not hasattr(processor, "_catalog_processor_class"):
            processor._catalog_processor_class = type(
                base.__name__, (_VariantBinding, base), {"__module__": processor.__name__}
            )
        codec = processor._catalog_processor_class.from_pretrained(directory, action_space=action_space)
        codec._variant_origin = origin
        codec._source_directory = root
        return codec
