# Supported tokenizer variants

All **34 variants** share `ZibinDong/ActionCodec2` at `revision="v0.1.0"`.
Pass the exact name below as `variant=`.

```text
{edition}-o{order}-{precision}-{vocabulary}-{gripper}-{speed}
```

`o1` / `o2`: encoding order; `5mm`: translation quantization; `4k`: 4,096 tokens
per profile; `2bin` / `21bin`: gripper levels; `1p2ms` / `2ms`: 1.2 / 2 m/s
EEF speed limit **per translation axis**, on the canonical 15 Hz grid. High-dynamic
joint profiles also raise the per-joint limit from 5.25 to 7.875 rad/s.

The selected revision's [variants.json](https://huggingface.co/ZibinDong/ActionCodec2/blob/v0.1.0/variants.json)
records canonical names, compatibility aliases, profiles, and asset fingerprints.
Names describe tokenizer artifacts, not VLA policy checkpoints.

| Variant | Profiles | Previous name (alias) |
| --- | --- | --- |
| `paper-o1-5mm-4k-2bin-1p2ms` | EEF, joint | `paper-1st` |
| `paper-o2-5mm-4k-2bin-1p2ms` | EEF, joint | `paper-2nd` |
| `extended-o1-1mm-2k-2bin-1p2ms` | EEF | `extended-1st-1mm-2k` |
| `extended-o1-1mm-4k-2bin-1p2ms` | EEF, joint | `extended-1st-1mm-4k` |
| `extended-o1-1mm-8k-2bin-1p2ms` | EEF, joint | `extended-1st-1mm-8k` |
| `extended-o1-1mm-16k-2bin-1p2ms` | EEF, joint | `extended-1st-1mm-16k` |
| `extended-o1-5mm-1k-2bin-1p2ms` | EEF, joint | `extended-1st-5mm-1k` |
| `extended-o1-5mm-2k-2bin-1p2ms` | EEF, joint | `extended-1st-5mm-2k` |
| `extended-o1-5mm-4k-2bin-1p2ms` | EEF, joint | `extended-1st-5mm-4k` |
| `extended-o1-5mm-8k-2bin-1p2ms` | EEF, joint | `extended-1st-5mm-8k` |
| `extended-o1-5mm-16k-2bin-1p2ms` | EEF, joint | `extended-1st-5mm-16k` |
| `extended-o1-10mm-1k-2bin-1p2ms` | EEF, joint | `extended-1st-10mm-1k` |
| `extended-o1-10mm-2k-2bin-1p2ms` | EEF, joint | `extended-1st-10mm-2k` |
| `extended-o1-10mm-4k-2bin-1p2ms` | EEF, joint | `extended-1st-10mm-4k` |
| `extended-o1-10mm-8k-2bin-1p2ms` | EEF, joint | `extended-1st-10mm-8k` |
| `extended-o1-10mm-16k-2bin-1p2ms` | EEF, joint | `extended-1st-10mm-16k` |
| `extended-o2-1mm-2k-2bin-1p2ms` | EEF | `extended-2nd-1mm-2k` |
| `extended-o2-1mm-4k-2bin-1p2ms` | EEF, joint | `extended-2nd-1mm-4k` |
| `extended-o2-1mm-8k-2bin-1p2ms` | EEF, joint | `extended-2nd-1mm-8k` |
| `extended-o2-1mm-16k-2bin-1p2ms` | EEF, joint | `extended-2nd-1mm-16k` |
| `extended-o2-5mm-1k-2bin-1p2ms` | EEF, joint | `extended-2nd-5mm-1k` |
| `extended-o2-5mm-2k-2bin-1p2ms` | EEF, joint | `extended-2nd-5mm-2k` |
| `extended-o2-5mm-4k-2bin-1p2ms` | EEF, joint | `extended-2nd-5mm-4k` |
| `extended-o2-5mm-8k-2bin-1p2ms` | EEF, joint | `extended-2nd-5mm-8k` |
| `extended-o2-5mm-16k-2bin-1p2ms` | EEF, joint | `extended-2nd-5mm-16k` |
| `extended-o2-10mm-1k-2bin-1p2ms` | EEF, joint | `extended-2nd-10mm-1k` |
| `extended-o2-10mm-2k-2bin-1p2ms` | EEF, joint | `extended-2nd-10mm-2k` |
| `extended-o2-10mm-4k-2bin-1p2ms` | EEF, joint | `extended-2nd-10mm-4k` |
| `extended-o2-10mm-8k-2bin-1p2ms` | EEF, joint | `extended-2nd-10mm-8k` |
| `extended-o2-10mm-16k-2bin-1p2ms` | EEF, joint | `extended-2nd-10mm-16k` |
| `sparse-o1-5mm-4k-2bin-1p2ms` | EEF, joint | `sparse-standard-2bin` |
| `sparse-o1-5mm-4k-21bin-1p2ms` | EEF, joint | `sparse-standard-21bin` |
| `sparse-o1-5mm-4k-2bin-2ms` | EEF, joint | `sparse-high-dynamic-2bin` |
| `sparse-o1-5mm-4k-21bin-2ms` | EEF, joint | `sparse-high-dynamic-21bin` |

Old names remain accepted by the updated runtime; no vocabulary or token IDs change.
Older pinned runtime code may require the previous name.

- Paper: the paper's tokenizers. Extended: retrained after more pretraining-data cleaning/filtering.
- Precision is not a guaranteed trajectory error bound; rotation/joint limits are in profile configs.
- Second order requires previous first-order increments as codec boundaries and model inputs.
- Sparse is experimental; all four releases are first order. See the [integration Skill](SKILL.md).
- G0.5 supports 4,096 tokens per profile; its Sparse adapter needs an extra EOS slot.
  Released Sparse profiles therefore require a smaller fitted vocabulary or an expanded head.

Update this table and revision from `variants.json` when publishing a new catalog.
