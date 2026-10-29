# Layouts and preprocessing

AC2 and Sparse share physical models, layouts, boundary handling and preprocessing.
The 13 built-in layouts use metres for EEF translation and radians for motion angles.

| Layout family | Columns per arm | Modes |
| --- | --- | --- |
| `single_eef_*`, `dual_eef_*` | xyz, rotation vector xyz, gripper | `abs`, `delta` |
| `single_joint6_*`, `dual_joint6_*` | 6 joints, gripper | `abs`, `delta` |
| `single_joint7_*`, `dual_joint7_*` | 7 joints, gripper | `abs`, `delta` |
| `single_joint6_abs_no_gripper` | 6 joints; no gripper | `abs` |

Dual-arm input columns contain the left arm followed by the right arm. Serialized
tokens advance whichever arm has the earliest unfinished frame; BPE tokens can
cover multiple frames. A token never combines two arms. Regenerate dual-arm
token caches produced by the older arm-by-arm format; single-arm tokens are unchanged. For six-joint arms, the
gripper maps to joint-profile slot 7; slot 6 is inactive, not an extra joint.

The old `single_joint6_abs_continuous_gripper` name remains an alias for the
six-coordinate layout without a gripper slot. It does not provide continuous
gripper control. For a continuous gripper, use a layout with a gripper column
and a compatible multi-level profile, such as a Sparse `21bin` variant.

EEF delta rotations compose on the right. Absolute layouts require the observed
`current_state`; second-order codecs also require `previous_first_order` in
canonical 15 Hz increments. Inspect custom conventions with `codec.print_action_spaces()`.

Filtering is disabled unless explicitly requested during tokenizer fitting:

```python
from actioncodec2 import PreprocessingSpec

codec.fit(
    episodes, fps=15, apply_preprocessing=True,
    preprocessing={"eef": PreprocessingSpec(
        profile="eef", trim_mode="off", savgol_window=9, savgol_polyorder=3,
    )},
)
```

Split recording discontinuities into continuous episodes before fitting.
Second-order cleaning additionally requires `bootstrap_context=True` and two
observed context frames. EEF smoothing leaves rotation vectors unchanged.
`codec.preprocess()` accepts a canonical single-arm trajectory for explicit data
preparation; it does not accept raw controller deltas. Saved preprocessing
settings never enable filtering in `encode`, `decode` or VLA training/inference.
