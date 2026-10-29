---
name: integrate-actioncodec2
description: Inspect a robotics or VLA project and recommend an ActionCodec2 integration and artifact selection. Implement the migration only after the user approves the proposal. Works without an existing ActionCodec2 installation or checkout.
---

# Integrate ActionCodec2

Follow two phases: **inspect and recommend**, then **implement after approval**.
The first deliverable is a concrete integration proposal, not code changes. During
assessment, do not install packages, download model weights, edit project files,
or start training. If the user has already approved the specific proposal, proceed
without asking again.

API baseline: **v0.1.0**. Use matching source/artifact revisions. These references
work from any project directory; no AC2 checkout is required:

- [Source](https://github.com/ZibinDong/actioncodec2/tree/v0.1.0)
- [Supported variants](https://github.com/ZibinDong/actioncodec2/blob/main/docs/variants.md) (`docs/variants.md` in a checkout)
- [Artifact catalog](https://huggingface.co/ZibinDong/ActionCodec2/tree/v0.1.0)
- [Layouts and preprocessing](https://github.com/ZibinDong/actioncodec2/blob/v0.1.0/docs/tokenizers.md)
- [G0.5 recipes](https://github.com/ZibinDong/actioncodec2/blob/v0.1.0/configs/README.md)
- [Policy evaluation](https://github.com/ZibinDong/actioncodec2/blob/v0.1.0/docs/evaluation.md)

## Phase A: inspect and recommend

### 1. Inspect the target project

Read project instructions, dependencies, dataset transforms, checkpoint settings,
model head, training loss, and inference loop. Establish:

| Contract | Required facts |
| --- | --- |
| Actions | EEF/joints, arm count, joint count, column order, absolute/delta, physical units |
| Rotations and gripper | Representation, frame, multiplication side; gripper range, polarity, binary/multi-level |
| Timing | Source frequency, action horizon, execution stride, episode boundaries |
| Model | Existing implementation, discrete/continuous head, token capacity, EOS/padding, previous-action input |
| Compatibility | Existing tokenizer identity, normalization, checkpoints, token caches |

Resolve facts from code and sample metadata first. When clarification is needed,
state the evidence, your best-supported interpretation, and a recommended option;
ask the user to confirm or choose a concrete alternative. Example:

> The loader scales xyz by 0.05 and labels actions as deltas. I recommend treating
> these as EEF delta commands with 0.05 metres per stored unit. Confirm this
> interpretation (Recommended), or specify the controller's different convention.

Do not invent a value when evidence is absent. Recommend obtaining the missing
controller/dataset specification before conversion rather than guessing from
array width. Ask only questions that affect the proposed integration.

### 2. Choose the model path

- **User already has a model:** retain it. Recommend downloading only the AC2
  tokenizer artifact and any missing runtime dependencies. Do not install G0.5,
  download its weights, or replace their training pipeline.
- **User has no model implementation:** recommend G0.5 as an available starting
  point, explaining its separate community license and training requirements.
- **Existing trained policy:** preserve its bundled tokenizer and conventions.
  Changing vocabulary changes token meanings and generally requires retraining.
- **Continuous/diffusion/flow head:** explain that AC2 output requires an
  autoregressive token head and training changes; include these in the proposal.

### 3. Choose an artifact, in this order

Use the shared repository **`ZibinDong/ActionCodec2`**, selected with `variant=`.

1. For checkpoint compatibility, keep the checkpoint's exact tokenizer.
2. For paper reproduction, use the matching `paper-o1-5mm-4k-2bin-1p2ms` or `paper-o2-5mm-4k-2bin-1p2ms`.
3. Otherwise recommend **`extended-o1-5mm-4k-2bin-1p2ms`** as the initial dense baseline.
4. Change precision, vocabulary budget, order, or Sparse mode only for an identified
   requirement. Read the supported variants list above, then confirm the combination
   and profiles in the selected Hub revision's `variants.json`, linked from that list.

Paper variants are the paper's tokenizers: 5 mm translation quantization and 4,096
tokens per profile. Extended variants were retrained after additional pretraining
data cleaning/filtering and provide more precision and vocabulary options.
Precision labels are quantization settings, not guaranteed trajectory error bounds.
The 1 mm / 2k dense variants contain EEF only. Equal budget/precision does not imply
identical token IDs.

**Second-order decision — explain before recommending it:**

Second order encodes the trajectory's second differences: changes in first-order
motion increments, using the codec's rotation geometry for orientations. The model
must receive the preceding timestep's first-order increment, and the codec needs
that same boundary for encoding and decoding. This may require adding a model
input and changing training/inference state handling. Recommend noise augmentation
of that conditioning input during training to improve robustness to deployment
boundary errors; propose its scale and application explicitly, without silently
changing existing experiments.

Ask: "First order avoids these changes (Recommended). Alternatively, accept second
order with previous-increment conditioning and the proposed training noise?"
Recommend second order only after the user accepts these requirements, or confirms
that their existing implementation already satisfies them. Otherwise use first order.

**Sparse decision — experimental, explicit acceptance required:**

Explain the maintainers' reported result: Sparse improves average compression by
**20–30% on the pretraining dataset**, with unchanged tracking precision. However,
limited downstream tests have performed worse than the standard dense version.
It remains experimental; unchanged codec precision does not guarantee unchanged
policy performance. Do not present the reported gain as a measurement on the
user's data or promise that performance cannot regress.

Ask: "Use the standard dense version (Recommended), or accept experimental Sparse
and validate downstream performance?" If accepted, select:

| Gripper | Standard motion | Higher dynamic range |
| --- | --- | --- |
| Binary | `sparse-o1-5mm-4k-2bin-1p2ms` | `sparse-o1-5mm-4k-2bin-2ms` |
| 21 levels | `sparse-o1-5mm-4k-21bin-1p2ms` | `sparse-o1-5mm-4k-21bin-2ms` |

Choose high-dynamic only when the observed motion needs its profile bounds. All
four released Sparse variants are first order and contain EEF and joint profiles.

### 4. Deliver the proposal and stop

Give a short proposal in the user's language with these five items:

1. **Project fit:** confirmed action/model contract and any unresolved assumption.
2. **Recommendation:** exact artifact, variant, revision, layout, and reason.
3. **Required changes:** dependencies, conversion adapter, head/token capacity,
   boundary input/noise if second order, and any retraining or cache regeneration.
4. **Validation:** representative round trip, one model batch, and offline reload.
5. **Decision:** any second-order/Sparse acceptance still needed.

End with: **"Would you like me to implement this migration? Proceed with the
proposed integration (Recommended), or revise the selection first."** Do not modify
code until approval. A variant preference alone is not approval to migrate.

## Phase B: implement the approved proposal

### 5. Install the selected path

Use the project's environment manager; inspect version conflicts before changing
its lockfile. Python 3.12 is the tested baseline. For an existing user model, the
Hub artifact carries the tokenizer runtime; install only missing dependencies:

```bash
python -m pip install numpy scipy torch 'transformers>=5.2,<6' 'huggingface-hub>=1.0' pyyaml
```

No AC2 checkout, CUDA, or LeRobot is required for this tokenizer-only path. Optional
C++17 kernels can later be built with `python -m pip install ./artifacts/my-ac2`
from a saved artifact. Keep incompatible dependency stacks in separate environments.

Only for an approved G0.5/source-package path, use Python >=3.12 and a C++17 compiler:

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/ZibinDong/actioncodec2.git
cd actioncodec2
python -m pip install -e .
```

Follow the linked G0.5 recipes/evaluation instructions. Its six released policies
are `ZibinDong/G05_ActionCodec2-{1st,2nd}_{LIBERO,SimplerEnv,DROID}`; each bundles
its tokenizer. Training from `OpenGalaxea/G05` requires gated access; released
policy inference does not need that base download. Preserve configured tracking
uploads and obtain agreement on experiment changes before training.

Honor the user's Hub endpoint/cache settings. When hf-mirror is requested, set
`HF_ENDPOINT=https://hf-mirror.com` before importing Hub libraries. Public AC2
artifacts need no login. Pin reviewed artifact code when using
`trust_remote_code=True`; never save credentials or machine-specific cache paths.

### 6. Implement the action boundary and model wiring

Start with this first-order example, then substitute the approved variant, layout,
converted target actions, and actual source frequency:

```python
import numpy as np
from transformers import AutoProcessor

codec = AutoProcessor.from_pretrained(
    "ZibinDong/ActionCodec2", revision="v0.1.0",
    variant="extended-o1-5mm-4k-2bin-1p2ms", action_space="single_eef_delta",
    trust_remote_code=True,
)
actions = np.zeros((1, 16, 7), dtype=np.float32)
actions[..., 0] = 0.005
actions[..., 6] = 1.0
rows = codec.encode(actions, fps=15)
restored = codec.decode(rows, fps=15)
assert restored.shape == actions.shape
assert np.isfinite(restored.numpy()).all()
```

Dense decode infers the horizon. Sparse requires
`codec.decode(rows, horizon=actions.shape[1], fps=15)`. Inputs are `(T,D)` or
`(B,T,D)`; decoded output is a CPU tensor `(B,T,D)`. Token rows have variable length.

Apply these contracts once, in an adapter shared by training and inference:

| Contract | Implementation |
| --- | --- |
| Layout | `single/dual_eef_abs/delta`, `single/dual_joint6_abs/delta`, `single/dual_joint7_abs/delta`; plus `single_joint6_abs_no_gripper` |
| Columns | EEF: xyz, rotation-vector xyz, gripper; joints: joint angles, gripper. Dual arms: left then right |
| Units | Metres, radians; undo controller normalization before encoding and invert after decoding |
| Rotation | Built-in EEF deltas compose on the right: `R_next = R_previous @ Exp(delta)`; convert other conventions explicitly |
| Gripper | Built-in layouts use absolute open fractions: 0 closed, 1 open; preserve existing checkpoint adapter conventions |
| Six joints | Gripper occupies physical slot 7; slot 6 is inactive. No extra input column. The old `continuous_gripper` alias actually has no gripper |
| Timing | Pass source `fps`; internal profiles use 15 Hz. Do not change target rate to bypass conversion |
| Absolute layouts | Pass observed pre-chunk `current_state` of shape `(B,D)` to both calls; omit it for delta layouts |
| Second order | Pass the same `previous_first_order` mapping in both calls and condition the model on the corresponding increment |

Inspect `codec.print_action_spaces()` for exact layouts. For second order,
`first_order_template(B)` gives component names/shapes, not a justified zero
boundary. Carry reconstructed boundaries with `final_first_order` after the
**executed prefix**, resetting at episode boundaries; `executed_steps` uses the
canonical 15 Hz grid. Do not use the unexecuted plan tail or source-frame counts.

Let the codec schedule dual-arm tokens by earliest unfinished frame; BPE tokens
can span frames but cannot combine arms. Older arm-by-arm caches must be regenerated;
affected token-sequence checkpoints require retraining.

For an autoregressive model, reserve IDs using the actual `codec.vocab_size`, not
just the per-profile 4k label. Keep EOS/padding outside the action interval and
strip them before codec decode. Preserve loss masks. Use `codec.grammar(horizon,
fps=...)` or `codec.logits_processor(...)` for constrained generation, and
`codec.max_tokens_for_horizon(...)` plus EOS space for generation limits. Streaming
must emit only covered frames; `decode_prefix` steps use the canonical grid.

G0.5-specific limits: dense adapters support single-arm EEF/seven-joint deltas;
Sparse additionally supports six-joint deltas. G0.5 reserves 4,096 rows per profile,
but its Sparse adapter needs one extra EOS, so a released 4,096-token Sparse profile
requires a smaller fitted vocabulary or an expanded, retrained head. Never truncate
its vocabulary. The Sparse factory uses `name: actioncodec2-sparse` and a standalone
saved artifact, not `variant=`. Other models should use the public codec directly.

### 7. Preserve training semantics and validate

Keep filtering **off in encode/decode and VLA training/inference**. Tokenizer fitting
also defaults to off: `codec.fit(episodes, fps=source_fps)`. Only approved tokenizer
pretraining may enable `apply_preprocessing=True` with explicit cleaning settings.
Second-order cleaning needs `bootstrap_context=True` and two context frames. Use
the linked preprocessing reference; never apply canonical-trajectory cleaning to
raw controller deltas. Fitting changes vocabulary identity: save separately,
regenerate caches, and train with those exact assets.

Before reporting completion:

1. Validate real moving/static samples, gripper transitions, both arms when present,
   and episode/partial-execution boundaries. Measure translation, rotation, joint,
   and gripper reconstruction errors as applicable; reconstruction is lossy.
2. Check shape, finite outputs, grammar completion, and identical training/inference
   tokenization. For an existing compatible policy, compare against its old tokens,
   decoded actions, and grammar masks.
3. Save with `codec.save_pretrained("./artifacts/my-ac2")`; reload using
   `AutoProcessor.from_pretrained(local_path, trust_remote_code=True,
   local_files_only=True)` with networking disabled. Keep the full runtime/assets.
4. Run one target training batch or inference call, checking finite loss/output,
   token count, and latency. Do not launch a full experiment without authorization.

Record repository, resolved revision, variant/fingerprint, and action conventions
with the policy. For a newly fitted artifact, remove stale catalog variant bindings.
Report changed files, the run command, validation results, and remaining training
or environment evaluation. Do not equate a codec round trip with policy success.
