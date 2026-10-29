# ActionCodec2

Official implementation of **ActionCodec2: A Streaming Action Tokenizer**.

![arXiv](https://img.shields.io/badge/arXiv-b31b1b)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E)](https://huggingface.co/collections/ZibinDong/actioncodec2-6ab6c4860ee90a881dda0a8b)
![Project Page](https://img.shields.io/badge/Project%20Page-4285F4)
![Paper](https://img.shields.io/badge/Paper-34A853)

- **Paper edition**: first- and second-order tokenizers used in the paper, with 5 mm translation precision and 4,096 tokens per profile.
- **Extended edition**: retrained after additional cleaning and filtering of the pretraining data, with 1/5/10 mm translation precision and 1k–16k vocabulary options.
- **Sparse version**: standard and high-dynamic ActionCodec2Sparse tokenizers for EEF and joints, with binary or 21-level grippers.

Integrating into your own project? Give your agent this [integration Skill](docs/SKILL.md).
It recommends a variant and migration plan, then implements it after your approval.

## Install

Requires Python 3.12 and a C++17 compiler.

```bash
git clone https://github.com/ZibinDong/actioncodec2.git && cd actioncodec2
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[cuda]'
```

## Evaluate the paper's VLA policies

| Dataset | First order | Second order |
| --- | --- | --- |
| LIBERO | `ZibinDong/G05_ActionCodec2-1st_LIBERO` | `ZibinDong/G05_ActionCodec2-2nd_LIBERO` |
| Bridge / SimplerEnv | `ZibinDong/G05_ActionCodec2-1st_SimplerEnv` | `ZibinDong/G05_ActionCodec2-2nd_SimplerEnv` |
| DROID | `ZibinDong/G05_ActionCodec2-1st_DROID` | `ZibinDong/G05_ActionCodec2-2nd_DROID` |

Start a policy server:

```bash
python scripts/serve.py --model ZibinDong/G05_ActionCodec2-2nd_LIBERO --trust-remote-code
```

In a separate [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) environment
with `numpy scipy websockets` installed, run from this repository's root:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  python scripts/benchmark_client.py --benchmark libero --benchmark-path /path/to/LIBERO \
      --suite "$suite" --episodes 50 || break
done
```

See [evaluation instructions](docs/evaluation.md) for LIBERO-Plus and SimplerEnv.

## Use tokenizers

Choose from the [complete variant list](docs/variants.md) with `variant`, for example `paper-o1-5mm-4k-2bin-1p2ms`, `extended-o2-5mm-4k-2bin-1p2ms`, or `sparse-o1-5mm-4k-2bin-1p2ms`.

```python
import numpy as np
from transformers import AutoProcessor

codec = AutoProcessor.from_pretrained(
    "ZibinDong/ActionCodec2",
    variant="paper-o1-5mm-4k-2bin-1p2ms",
    action_space="single_eef_delta",
    trust_remote_code=True,
)
actions = np.zeros((1, 16, 7), dtype=np.float32)
tokens = codec.encode(actions, fps=15)
reconstructed = codec.decode(tokens, fps=15)
```

This layout uses translation deltas in metres, right-multiplied rotation-vector deltas
in radians, and a gripper command in `[0, 1]`. Set `fps` to your data's rate.
Second-order variants also require `previous_first_order` (canonical 15 Hz increments
by component) in both calls. See the [tokenizer reference](https://huggingface.co/ZibinDong/ActionCodec2#tokenizer-usage) for other layouts and boundaries.

## Train tokenizers

Using the codec above, fit a vocabulary on your own episodes with shape `(N, T, 7)`:

```python
episodes = np.load("episodes.npy", allow_pickle=False)
codec.fit(episodes, fps=15)
codec.save_pretrained("./my-tokenizer")
```

Cleaning is off by default. Enable it only for tokenizer pretraining with
`apply_preprocessing=True` and an explicit `preprocessing` configuration; see the
[layout and preprocessing reference](docs/tokenizers.md). Sparse decoding requires `horizon`.

Set `tokenizer.artifact` to `./my-tokenizer` and remove `variant` and
`variant_fingerprint` from the VLA recipe to train with it.

## Train VLA policies

Prepare datasets following [configs/README.md](configs/README.md) and request access
to [OpenGalaxea/G05](https://huggingface.co/OpenGalaxea/G05), then run:

```bash
hf auth login
python scripts/train.py --config configs/train/libero/actioncodec2.yaml --data-root /path/to/datasets
```

Use `actioncodec2_second_order.yaml` for second order; Bridge and DROID recipes are
in `configs/train/bridge/` and `configs/train/droid/`. For multiple GPUs, replace
`python` with `torchrun --nproc-per-node 8`.

## License

Code: [Apache 2.0](LICENSE), except the G0.5 adaptation and derived policies, which
use the non-commercial [G0.5 Community License](LICENSE-G0.5). See [NOTICE](NOTICE)
for third-party licenses.
