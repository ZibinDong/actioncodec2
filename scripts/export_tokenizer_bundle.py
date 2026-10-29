"""Export an ActionCodec2 processor with portable C++ acceleration sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from actioncodec2.publishing import prepare_public_assets  # noqa: E402

CORE_FILES = ("config.yaml", "primitives.npz", "bpe.json", "config.json", "processor_config.json")


def model_card(repo_id: str, *, edition: str, order: int,
               precision: str | None = None, codebook_size: int = 4096,
               profile_names: tuple[str, ...] = ("joint", "eef"),
               layout_count: int = 13,
               collections: dict[str, str] | None = None) -> str:
    """Describe the exported artifact's order, profiles and boundary contract."""
    if order not in (1, 2) or edition not in ("Paper", "Extended"):
        raise ValueError("expected Paper/Extended edition and order 1/2")
    if edition == "Extended" and precision not in ("1mm", "5mm", "10mm"):
        raise ValueError("extended artifacts require a precision tier")
    if edition == "Paper":
        precision = "5mm"
    ordinal = "First" if order == 1 else "Second"
    suffix = f" — {precision}, {codebook_size // 1024}k" if edition == "Extended" else ""
    description = ("Tokenizer used in **ActionCodec2: A Streaming Action Tokenizer**, "
                   "with 5 mm translation precision and a 4,096-token budget per profile."
                   if edition == "Paper" else
                   "Extended ActionCodec2 tokenizer retrained after additional cleaning and "
                   "filtering of the pretraining data. The Extended edition provides more "
                   "translation-precision and vocabulary-budget options.")
    catalog = collections or {
        "All tokenizers": "https://hf-mirror.com/collections/ZibinDong/actioncodec2-6ab6c4860ee90a881dda0a8b",
    }
    navigation = " · ".join(f"[{name}]({url})" for name, url in catalog.items())
    boundary_setup = ("# This synthetic episode starts from rest.\n"
                      "boundary = codec.first_order_template(batch_size=1)\n" if order == 2 else "")
    boundary_arg = ", previous_first_order=boundary" if order == 2 else ""
    boundary_notes = """
Second-order motion requires `previous_first_order` at every chunk boundary.
Use a zero template only when the preceding motion is known to be zero.
For the next chunk, carry the reconstructed boundary:

```python
boundary = codec.final_first_order(tokens, previous_first_order=boundary, executed_steps=None)
```

For a partial chunk, `executed_steps` counts steps at the tokenizer's internal 15 Hz,
even when `encode` or `decode` uses a different `fps`. Supply boundaries in that same
15 Hz convention. The VLA adapter handles conversion from the last raw executed action.
""" if order == 2 else ""
    return f"""---
license: apache-2.0
library_name: transformers
tags:
- robotics
- action-tokenization
- actioncodec2
- {edition.lower()}-edition
- {order}{'st' if order == 1 else 'nd'}-order
---

# ActionCodec2 {edition} {ordinal}-Order{suffix}

{description}

{navigation}

| Setting | Value |
| --- | --- |
| Motion order | {order} |
| Translation precision | {precision} |
| Codebook budget | {codebook_size} per profile |
| Fitted profiles | {', '.join(profile_names)} |
| Registered layouts | {layout_count} |
| Target rate | 15 Hz |
| Gripper | Binary, zero-order; open when ≥ 0.8 |

The budget applies separately to each profile; joint and EEF tokens use disjoint namespaces.
Translation precision describes the quantization setting; rotation and joint settings are
specified separately in each profile's configuration. Reconstruction is lossy, and policy
accuracy also depends on the policy and task.

## Usage

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install numpy scipy torch "transformers>=4.57,<6" huggingface-hub pyyaml
```

```python
import numpy as np
from transformers import AutoProcessor

codec = AutoProcessor.from_pretrained("{repo_id}", trust_remote_code=True)
actions = np.zeros((12, 7), dtype=np.float32)
actions[:, 6] = 1.0
{boundary_setup}
tokens = codec.encode(actions, fps=15{boundary_arg})
decoded_actions = codec.decode(tokens, fps=15{boundary_arg})
```
{boundary_notes}
The default layout is `single_eef_delta`: position increments in metres (columns 0–2),
body-frame rotation-vector increments in radians (3–5), and an absolute gripper command (6).
Rotation follows `R_next = R_previous @ Exp(rotvec)`. Supply physical values and the actual
recording rate through `fps`; the tokenizer resamples to 15 Hz.

Inputs may have shape `(T, D)` or `(B, T, D)`. Decoding returns a CPU float32 tensor
of shape `(B, T_out, D)`; resampling can change the length. Quantization is lossy.

Inspect and select layouts explicitly:

```python
codec.print_action_spaces()
codec = codec.for_action_space("single_eef_abs")
```

Absolute layouts also need `current_state` for absolute reconstruction.
{'Joint, EEF, single-arm and dual-arm layouts are included.' if 'joint' in profile_names else 'This release contains EEF layouts only, for single and dual arms.'}

## Native acceleration

Build the optional C++17 extension to accelerate encoding and decoding:

```bash
hf download {repo_id} --local-dir ./actioncodec2-tokenizer
pip install ./actioncodec2-tokenizer
```

A C++17 compiler is required. The processor automatically uses the installed
kernels; installing the [ActionCodec2 source](https://hf-mirror.com/ZibinDong/ActionCodec2/blob/main/README.md#vla-inference-package)
also supplies them. Without kernels, it uses the bundled Python runtime.
`ACTIONCODEC2_NUM_THREADS` controls the per-process thread budget;
`ACTIONCODEC2_NO_NATIVE=1` selects the Python reference implementation.

Keep the complete repository when downloading or saving: `router_config.yaml`, `profiles/`,
`runtime/`, the Hugging Face entry point, metadata, native sources and license.
"""


def export_bundle(source: Path, output: Path, *, profiles: dict[str, Path] | None = None,
                  codebook_size: int | None = None) -> None:
    """Export fitted assets with the installed shared runtime."""
    from actioncodec2.core.catalog import ActionCodec2
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if source == output or source in output.parents:
        raise ValueError("output must be outside the source artifact")
    if (source / "variants.json").is_file():
        if profiles is not None or codebook_size is not None:
            raise ValueError("catalog refresh cannot replace fitted profiles")
        export_catalog(source, output, repo_id="ZibinDong/ActionCodec2")
        return
    codec = ActionCodec2.from_pretrained(source, local_files_only=True)
    if profiles is not None:
        from actioncodec2.core.backends.actioncodec2 import ActionCodec2Processor, ActionCodec2Router
        if not profiles or set(profiles) - {"eef", "joint"}:
            raise ValueError("profiles must contain eef and/or joint")
        processors = {name: ActionCodec2Processor.from_pretrained(path) for name, path in profiles.items()}
        router = ActionCodec2Router.from_processors(
            processors, token_budgets={name: codebook_size or p.vocab_size for name, p in processors.items()},
            presets=[p for p in codec.action_spaces.values() if {c.space for c in p.components} <= processors.keys()],
        )
        codec = type(codec)(router, action_space=codec.action_space.name)
        codec._variant_origin = {}
    codec.save_pretrained(output)
    prepare_public_assets(output)


def catalog_model_card(variants: dict, repo_id: str) -> str:
    """Document selectable artifacts and the shared input contract."""
    rows = "\n".join(
        f"| `{name}` | {r['edition']} | {r['encoding_order']} | {r['precision']} | {r['codebook_size']} | {', '.join(r['profiles'])} |"
        for name, r in variants.items()
    )
    return f'''---
license: apache-2.0
library_name: transformers
tags:
- robotics
- action-tokenization
- actioncodec2
---

# ActionCodec2

- **Paper edition**: first- and second-order tokenizers used in the paper; 5 mm translation precision and 4,096 tokens per profile.
- **Extended edition**: retrained after additional pretraining-data cleaning and filtering, with more precision and vocabulary options.
- **Sparse version**: standard and high-dynamic first-order tokenizers, each with EEF and joint profiles and binary (`2bin`) or 21-level (`21bin`) grippers.

All {len(variants)} variants use one shared runtime. Precision labels describe translation quantization; rotation and joint limits are in each profile's configuration.

Names encode edition, order (`o1`/`o2`), translation precision, vocabulary budget,
gripper levels, and EEF speed per translation axis: `1p2ms` = 1.2 m/s, `2ms` = 2 m/s.
Previous names remain compatibility aliases in `variants.json`; token IDs are unchanged.

## Tokenizer usage

```bash
pip install numpy scipy torch "transformers>=5.2,<6" huggingface-hub pyyaml
```

```python
import numpy as np
from transformers import AutoProcessor

codec = AutoProcessor.from_pretrained(
    "{repo_id}", variant="paper-o1-5mm-4k-2bin-1p2ms",
    action_space="single_eef_delta", trust_remote_code=True,
)
actions = np.zeros((1, 16, 7), dtype=np.float32)
tokens = codec.encode(actions, fps=15)
restored = codec.decode(tokens, fps=15)
```

Use `variant="sparse-o1-5mm-4k-2bin-1p2ms"` or `"sparse-o1-5mm-4k-21bin-2ms"` for Sparse;
Sparse decoding requires `horizon=actions.shape[-2]` in source frames.
Each profile has its own token namespace; the budget is per profile, not the total routed vocabulary.

The EEF layout is `[x, y, z, rx, ry, rz, gripper]`, in metres and radians.
Delta rotations use `R_next = R_previous @ Exp(rotvec)`. Grippers are absolute open fractions (0 closed, 1 open).
Inputs are `(T,D)` or `(B,T,D)` and are resampled to 15 Hz. Reconstruction is lossy.
Absolute layouts also require `current_state`; second-order variants require
`previous_first_order` in canonical 15 Hz increments in both encode and decode.
Zero boundaries are valid only when the preceding motion is known to be zero.
Use `final_first_order` to carry reconstructed second-order boundaries between chunks.

`codec.print_action_spaces()` lists the layouts. Dual arms are interleaved by time;
BPE tokens never merge across arms. The historical `single_joint6_abs_continuous_gripper`
name aliases `single_joint6_abs_no_gripper`: six joint coordinates, no separate gripper slot.

## Train tokenizers

```python
codec.fit(episodes, fps=15)
codec.save_pretrained("./my-tokenizer")
```

Cleaning is off by default, including VLA training and inference. Only tokenizer fitting
accepts `apply_preprocessing=True`, with a recorded or explicitly supplied `preprocessing`
configuration. It acts on continuous canonical trajectories before quantization.
Split recording discontinuities first; second-order cleaning needs `bootstrap_context=True`.
Saved cleaning settings do not enable filtering during encode or decode.

## Available variants

| Variant | Edition | Order | Translation precision | Budget / profile | Profiles |
| --- | --- | --- | --- | --- | --- |
{rows}

Pin `revision` for reproducibility. `variants.json` records asset hashes; a selected
`save_pretrained()` directory contains its assets and the shared portable runtime.
The optional C++17 kernels can be built with `pip install ./downloaded-tokenizer`.

## VLA policies

Installation, paper-policy evaluation and VLA training: [source repository](https://github.com/ZibinDong/actioncodec2).
The existing six AC2 policies retain their single-arm tokenizer behavior. The released Sparse
profiles have a 4,096-token budget; the current G0.5 Sparse adapter additionally needs an EOS
position and therefore requires a vocabulary of at most 4,095 tokens when training a new VLA.

## License

Tokenizer code and assets: Apache-2.0. G0.5 policy code and weights use their separate community license.
'''


def export_catalog(source: Path, output: Path, *, repo_id: str,
                   sparse_sources: dict[str, Path] | None = None) -> None:
    """Assemble dense and sparse variants around one generated runtime."""
    from actioncodec2.core.catalog import ActionCodec2, VARIANT_ALIASES, asset_inventory, inventory_fingerprint
    from actioncodec2.core.exporting import save_runtime
    from actioncodec2 import ActionCodec2Sparse

    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if source == output or source in output.parents:
        raise ValueError("output must be outside the source catalog")
    entries = {}
    if (source / "variants.json").is_file():
        for name, record in json.loads((source / "variants.json").read_text())["variants"].items():
            entries[name] = (ActionCodec2.from_pretrained(source, variant=name), dict(record))
    else:
        for item in json.loads((source / "manifest.json").read_text()):
            directory = item["directory"]
            if Path(directory).name != directory or directory in (".", ".."):
                raise ValueError("release directory must be one directory name")
            codec = ActionCodec2.from_pretrained(source / directory)
            order = item.get("encoding_order", item.get("order"))
            edition = item["edition"]
            if order not in (1, 2) or edition not in ("Paper", "Extended"):
                raise ValueError("invalid dense release edition/order")
            budget = {r.spec.token_budget for r in codec.action_tokenizer.profiles.values()}
            if len(budget) != 1:
                raise ValueError("catalog variants require one per-profile budget")
            budget = budget.pop()
            precision = "5mm" if edition == "Paper" else item["precision"]
            name = f"paper-{'1st' if order == 1 else '2nd'}" if edition == "Paper" else f"extended-{'1st' if order == 1 else '2nd'}-{precision}-{budget // 1024}k"
            if name in entries:
                raise ValueError(f"duplicate variant: {name}")
            entries[name] = (codec, dict(edition=edition, encoding_order=order, precision=precision,
                codebook_size=budget, profiles=sorted(codec.fitted_profiles), legacy_repo_id=item["repo_id"]))
    for edition, package in (sparse_sources or {}).items():
        if edition not in {"standard", "high-dynamic"}:
            raise ValueError("sparse edition must be standard or high-dynamic")
        for mode in ("2bin", "21bin"):
            codec = ActionCodec2Sparse.from_pretrained(package, action_space="single_eef_delta",
                                                      primitive_order=1, gripper_mode=mode, verify_gates=True)
            name = f"sparse-{edition}-{mode}"
            entries[name] = (codec, dict(edition="Sparse", tokenizer="actioncodec2_sparse",
                dynamics=edition, encoding_order=1, precision="5mm", codebook_size=4096,
                gripper_mode=mode, profiles=sorted(codec.fitted_profiles)))
    output.mkdir(parents=True)
    variants = {}
    runtime_files = ("runtime", "actioncodec2_kernels", "setup.py", "pyproject.toml", "processing_actioncodec2.py", "LICENSE")
    for name, (codec, record) in entries.items():
        name = VARIANT_ALIASES.get(name, name)
        if name in variants:
            raise ValueError(f"duplicate canonical variant: {name}")
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
            raise ValueError("invalid variant name")
        target = output / "variants" / name
        codec.save_pretrained(target)
        for relative in (*runtime_files, "artifact_manifest.json", "README.md"):
            path = target / relative
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        record.pop("files", None)
        record.pop("fingerprint", None)
        variants[name] = record
    (output / "processor_config.json").write_text(json.dumps({
        "processor_class": "ActionCodec2", "auto_map": {"AutoProcessor": "processing_actioncodec2.ActionCodec2"},
        "variant_catalog": "variants.json",
    }, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({"model_type": "actioncodec2", "processor_class": "ActionCodec2"}, indent=2)+"\n")
    save_runtime(output)
    prepare_public_assets(output)
    for name, record in variants.items():
        files = asset_inventory(output / "variants" / name)
        record.update(files=files, fingerprint=inventory_fingerprint(files))
    (output / "variants.json").write_text(json.dumps({"format_version": 1, "variants": variants,
        "aliases": {old: new for old, new in VARIANT_ALIASES.items() if new in variants}}, indent=2)+"\n")
    (output / "README.md").write_text(catalog_model_card(variants, repo_id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eef-profile", type=Path)
    parser.add_argument("--joint-profile", type=Path)
    parser.add_argument("--codebook-size", type=int)
    parser.add_argument("--repo-id", help="write an English model card for this Hub repository")
    parser.add_argument("--edition", choices=("Paper", "Extended"), default="Paper")
    parser.add_argument("--order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--precision", choices=("1mm", "5mm", "10mm"))
    parser.add_argument("--sparse-standard", type=Path)
    parser.add_argument("--sparse-high-dynamic", type=Path)
    parser.add_argument("--catalog", action="store_true", help="merge the source release manifest")
    args = parser.parse_args()
    if args.catalog:
        if args.eef_profile or args.joint_profile or args.codebook_size or not args.repo_id:
            parser.error("--catalog requires --repo-id and cannot replace fitted profiles")
        export_catalog(args.source, args.output, repo_id=args.repo_id, sparse_sources={
            name: path for name, path in (("standard", args.sparse_standard), ("high-dynamic", args.sparse_high_dynamic)) if path is not None
        })
        return
    profiles = {name: path for name, path in (
        ("eef", args.eef_profile), ("joint", args.joint_profile),
    ) if path is not None}
    export_bundle(args.source, args.output, profiles=profiles or None,
                  codebook_size=args.codebook_size)
    if args.repo_id:
        router = yaml.safe_load((args.output / "router_config.yaml").read_text())
        sizes = {p["token_budget"] for p in router["profiles"]}
        if len(sizes) != 1:
            raise ValueError("model-card generation requires a common per-profile token budget")
        (args.output / "README.md").write_text(model_card(
            args.repo_id,
            edition=args.edition, order=args.order, precision=args.precision,
            codebook_size=sizes.pop(), profile_names=tuple(p["name"] for p in router["profiles"]),
            layout_count=len(router["presets"]),
        ))


if __name__ == "__main__":
    main()
