# Configuration

A run configuration has five sections: `model`, `tokenizer`, `data`, `train`,
and `tracking`. Fields are documented where they are defined, in
[`src/actioncodec2/vla/training/config.py`](../src/actioncodec2/vla/training/config.py);
unknown fields are rejected in every section, including `tokenizer`, where each
adapter accepts only the keys it reads.
`python scripts/train.py --config <recipe> --data-root <datasets> --print-config`
prints the recipe as loaded; `data.control_frequency_hz`, when a recipe leaves it
out, is taken from the dataset's `info.json` when training starts.

A recipe may inherit from a sibling file with `extends: base.yaml`; its sections
are merged one level deep. Paths starting with `./` or `../` resolve against the
repository root; a bare name such as `physical-intelligence/fast` is a Hugging
Face repository.

## Recipes

- `quickstart.yaml`: the quickstart from the top-level README, FAST on the
  public `lerobot/libero` dataset, using the committed LIBERO statistics
  directly, so no local fitting step is needed.
- `train/{libero,bridge,droid}/`: one `base.yaml` with the dataset, G0.5, and
  optimizer settings, and one file per tokenizer: `actioncodec2`,
  `actioncodec2_second_order`, `actioncodec`, `fast`, and `x_tokenizer`.
  Published VLA policies cover ActionCodec2 first and second order. Other tokenizer
  recipes remain available for training.
- `stats/`: normalization files required by the recipes. To fit your own, use
  `scripts/fit_lerobot_stats.py --validation-fraction <fraction>` with the same
  fraction and root order as training. Pass `0` explicitly when all episodes
  belong to training. Existing release statistics are not refitted by training.

```bash
python scripts/train.py --config configs/train/libero/actioncodec2.yaml --data-root /path/to/libero
```

## What a recipe needs from you

| Input | How | Default |
|---|---|---|
| Datasets | `--data-root <directory>`; each entry of `data.roots` is resolved against it | required |
| G0.5 base weights | `--base-checkpoint <model_state_dict.pt>` | downloaded from the gated `OpenGalaxea/G05` repository |
| Qwen3.5 text tokenizer | `--text-tokenizer <directory>` | the copy in `OpenGalaxea/G05` |

Everything else is either in the recipe or fetched from the Hub. ActionCodec2 uses
`ZibinDong/ActionCodec2` with a named variant and an asset fingerprint; the resolved
Hub commit is saved in the run configuration. ActionCodec, FAST, and X-Tokenizer
recipes pin their upstream revision. Other settings are overridden with `--steps`,
`--output-dir`, `--learning-rate`, `--save-every`, `--tracker`, and `--run-name`.

Dataset layouts the recipes expect under `--data-root`:

| Recipe | `data.roots` |
|---|---|
| `train/libero/*` | `libero_{10,goal,object,spatial}_no_noops_lerobot`, 20 Hz, cameras `image` and `wrist_image` |
| `train/bridge/*` | `bridge_orig_lerobot` |
| `train/droid/*` | `split0` … `split15` |
| `quickstart.yaml` | the `lerobot/libero` snapshot itself (`.`), 10 Hz, cameras `image` and `image2` |

Training checks frame rate and camera names against `info.json` before loading
the model. `scripts/fit_lerobot_stats.py` fits `proprio_stats.json` and
`action_stats.json` for a dataset of your own (`--state-key` and `--action-key`
for non-standard columns).

## Action tokenizers

`tokenizer.name` selects the adapter; the other keys are passed to it.

| `name` | Released as | Notes |
|---|---|---|
| `actioncodec2` | [`ZibinDong/ActionCodec2`](https://huggingface.co/ZibinDong/ActionCodec2), variants `paper-o1-5mm-4k-2bin-1p2ms` and `paper-o2-5mm-4k-2bin-1p2ms` | The second-order release encodes each chunk relative to the action before it (see below). Supports EEF and 8-D joint actions. |
| `actioncodec` | [`ZibinDong/ActionCodec-5e`](https://huggingface.co/ZibinDong/ActionCodec-5e) | Needs the `actioncodec` extra (`vector-quantize-pytorch`). |
| `fast` | [`physical-intelligence/fast`](https://huggingface.co/physical-intelligence/fast) | |
| `x-tokenizer` | [`x-square-robot/X-Tokenizer`](https://huggingface.co/x-square-robot/X-Tokenizer) | Needs the official Python package; install it only for these recipes: `uv pip install "x-tokenizer @ git+https://github.com/X-Square-Robot/X-Tokenizer.git@9733990b16a07d51fb8b5d305ad8c16abb57aef2"`. The LIBERO recipe uses `calibration_path`; omitting it selects the uncalibrated output. |

Set `tokenizer.artifact` to a Hub repository or a local artifact directory.
Artifacts declaring `input_order: 1` require `target_hz` equal to the dataset's
rate, `motion_scale: 1`, and matching rotation conventions. Export preserves
local assets and remote repository revisions.

For a merged ActionCodec2 release, set `tokenizer.artifact` to the catalog directory
or Hub repository and set `tokenizer.variant`, for example `paper-o1-5mm-4k-2bin-1p2ms`, `paper-o2-5mm-4k-2bin-1p2ms`,
or `extended-o2-5mm-4k-2bin-1p2ms`. Run configs record the selected asset fingerprint and,
for Hub catalogs, the resolved commit. A mismatching `variant_fingerprint` is rejected.
Policy export bundles the selected variant and its portable runtime, including when
the training configuration refers to a remote catalog. Published policies load their
own bundled assets and require no further tokenizer download.

`paper-o1-5mm-4k-2bin-1p2ms` and `paper-o2-5mm-4k-2bin-1p2ms` are the versions used in the paper, both at 5 mm translation
precision with a 4,096-token budget per profile. Extended variants were retrained after
additional cleaning and filtering of the pretraining data, with more precision and
vocabulary-budget options. Use the same variant for training and inference; matching
precision and budget do not imply identical token IDs.

G0.5 supports vocabularies of up to 4096 tokens per profile; 8k/16k ActionCodec2 weights
are for standalone use. ActionCodec2, ActionCodec and X-Tokenizer decode under
the tokenizer's grammar, so a chunk always parses. FAST is the exception in the
released recipes: they set `constrained_decoding: false`, which decodes FAST as it
is normally used, and a stream that does not parse becomes a zero action.
Set `constrained_decoding: true` to mask illegal ids.

## Dataset conventions

The `data` section states the physical meaning of the stored actions. The
tokenizer inherits `time_horizon`, `action_dim`, `fps`, `gripper_open_is_high`,
and (for end-effector actions) `rotation_representation` from it unless the
tokenizer section sets them, and the policy server reports the dataset's
conventions to simulator clients.

| | LIBERO | Bridge | DROID |
|---|---|---|---|
| actions | delta EEF, rotation vector | delta EEF, Euler XYZ | joint deltas derived from state (`action_offset: 1`) |
| control rate / horizon | 20 Hz / 20 | 5 Hz / 10 | 15 Hz / 15 |
| gripper | 1 = open | 1 = open | 1 = closed |
| proprio | observed | none | observed |

`data.translation_meters_per_unit` and `data.rotation_radians_per_unit` state what
one stored unit of an end-effector action means physically (LIBERO stores
normalized controller commands, 0.05 m and 0.5 rad per unit; Bridge stores meters
and radians). The policy server reports them to clients, and leaves them out when
a recipe does not set them.

`data.gripper_open_is_high` describes the dataset. A tokenizer may use the
opposite convention internally: the released LIBERO ActionCodec2 recipe sets
`tokenizer.gripper_open_is_high: false` although LIBERO stores 1 = open. Do not
"fix" one to match the other; a policy is only consistent with the tokenizer
setting it was trained with. `data.gripper_range` (default `[0, 1]`) states the
stored values at the gripper's extremes, and `lerobot/libero` uses `[-1, 1]`
with -1 = open (FAST only).

The released policies' ActionCodec2 recipes use the released EEF profile and
per-device batch settings; reduce `train.batch_size` and increase
`train.gradient_accumulation` for smaller GPUs. DROID uses joint deltas
(8-D). Its released policies cover ActionCodec2 (1st and 2nd order);
the ActionCodec recipe uses the released 7-D EEF embodiment,
because ActionCodec has no 8-D joint embodiment, and has no released policy.

## Stateful tokenizers

Second-order ActionCodec2 encodes a chunk relative to the action before it. The
`data.previous_action_source` setting chooses that boundary: `ground_truth` uses
the recorded preceding action, while `reconstructed` rolls the codec through the
episode at `data.execution_stride` (the number of actions a deployed policy
executes per plan). The token cache stores these boundaries too, so training
never re-runs the rollout.

`data.previous_action_augmentation` is optional and supported by ActionCodec2.
It requires `previous_action_source: ground_truth`, `action_offset: 0`, a
previous-action model input, and `token_cache: null`.
