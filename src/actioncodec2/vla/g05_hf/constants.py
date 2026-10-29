# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Layout constants of the released G0.5 vocabulary.

Kept apart from ``configuration_g05`` so that tools which only need the numbers
(such as the bundle exporter) do not import Transformers.
"""

# Token layout of the released G0.5 base vocabulary (Qwen3.5 text tokens followed
# by 4096 native action rows and a few special tokens).
G05_ACTION_BEGIN = 248_077
G05_ACTION_ROWS = 4096
G05_EOV_ID = 252_176
G05_STATE_ID = 252_177
G05_VOCAB_SIZE = 252_189
#: Width of the released proprio adapter input, laid out as
#: ``left_control[0:9] left_gripper[9] right_control[10:19] right_gripper[19]
#: lower_body[20:27]``.
G05_INTERNAL_PROPRIO_DIM = 27
#: ``model_type`` of :class:`~actioncodec2.vla.g05_hf.configuration_g05.G05Config`.
G05_MODEL_TYPE = "g05_action_policy"
#: ``model_type`` written by checkpoints saved before the type was renamed.
LEGACY_MODEL_TYPES = ("neo_one_step_vla",)
