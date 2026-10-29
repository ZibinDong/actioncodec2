# Policy WebSocket protocol

Requests use JSON: `{"id": 1, "method": "metadata", "params": {}}`.
Responses echo `id` and contain either `result` or `error`.
Error details are recorded in the server logs; responses contain a generic message.
Arrays use `{"dtype": "<f4", "shape": [20, 7], "data": "<base64 bytes>"}`.

| Method | Params | Result |
|---|---|---|
| `metadata` | None | Action specification, camera keys, image size, proprio mode, action horizon, execution horizon, and `requires_previous_action` |
| `health` | None | Metadata plus `ready: true` |
| `predict` | `frames`, `task`, optional `state`, `previous_action`, `temperature` | `actions`, `tokens`, `inference_seconds` |
| `predict_batch` | `observations` (list of predict params), optional `temperature` | `results`, `batch_size`, `inference_seconds` |

Read `metadata` before predicting:

- `frames`: one RGB `uint8` H×W×3 array per camera, in `camera_keys` order.
- `state`: required for policies using observed proprioception, with `state_dim`
  raw values. Send the complete observed vector; the processor selects and normalizes
  the model's dimensions. Released LIBERO policies take 8 raw values, including both fingers.
- `previous_action`: last executed action; required when `requires_previous_action` is true.
- `actions`: `[action_horizon, action_dim]` in the model's dataset conventions.
  Execute `execution_horizon` actions before requesting another plan.
- `action_spec`: action space, frame, rotation representation, control frequency,
  gripper convention, and translation/rotation unit scales when specified by the dataset.
