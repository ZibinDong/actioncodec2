# Evaluation

Run the policy server in this project's environment and the client in the benchmark's environment.
All commands run from this repository's root. The client reports task success rates.

## Policy server

```bash
python scripts/serve.py --model ZibinDong/G05_ActionCodec2-2nd_LIBERO --trust-remote-code
```

| Flag | Meaning |
|---|---|
| `--model` | Hugging Face model repo ID or local model directory (required) |
| `--device` | Inference device; default `cuda:0` |
| `--host`, `--port` | Listen address; default `127.0.0.1:18080` |
| `--trust-remote-code` | Allow bundled tokenizer code to run; required for Hub models |
| `--dtype` | Weight dtype: `float32` (default, with bfloat16 autocast) or `bfloat16` |
| `--max-batch-size` | Maximum requests decoded together; default 32 |
| `--batch-wait-ms` | Wait for requests to form a batch; default 3 ms |

Optional: `--revision <branch/tag/commit>` loads a specific version of the Hub model.
Omit it to use `main`.

The six published policies cover ActionCodec2 first and second order on LIBERO,
Bridge/SimplerEnv, and DROID. Each policy includes its action-tokenizer assets and
runtime; after downloading the complete repository, use a local `--model` directory
with `HF_HUB_OFFLINE=1`. No base-checkpoint download is needed for inference.
Use the chosen policy's `policy_config.json` for its camera, gripper, frequency,
and execution-stride contract; training recipes do not override a saved policy.

## Benchmark environments

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), or
[SimplerEnv](https://github.com/simpler-env/SimplerEnv), including its assets.
Use separate environments for LIBERO and LIBERO-Plus: both install the `libero` package.
In the benchmark environment, also install:

```bash
pip install numpy scipy "websockets>=14"
```

## Common client flags

| Flag | Meaning |
|---|---|
| `--benchmark` | `libero`, `libero-plus`, or `simplerenv` (required) |
| `--benchmark-path` | Benchmark repository checkout (required) |
| `--server` | Policy server URL; default `ws://127.0.0.1:18080` |
| `--episodes` | Episodes per task; default 1. LIBERO-Plus requires 1 |
| `--episode-start` | First episode index; default 0. LIBERO-Plus requires 0 |
| `--seed` | Simulator random seed; default 7 |
| `--max-steps` | Step limit per episode; defaults are listed below |
| `--settle-steps` | Initial no-op steps; default 10 |

## LIBERO

Four suites × 10 tasks × 50 episodes = **2,000 episodes**.

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  python scripts/benchmark_client.py --benchmark libero --benchmark-path /path/to/LIBERO \
      --suite "$suite" --episodes 50 || break
done
```

| Flag | Meaning |
|---|---|
| `--suite` | `libero_spatial` (default), `libero_object`, `libero_goal`, or `libero_10` |
| `--task-id` | Task index in the suite; omit to run all tasks |
| `--libero-config` | Directory containing `config.yaml`; configured automatically if omitted |

Episodes use the official initial states in order, starting at `--episode-start`.
Default step limits: Spatial/Object 280, Goal 300, 10 520.
Control frequency follows the model; use the 20 Hz LIBERO recipes for these examples.

## LIBERO-Plus

Evaluate every perturbed task once, using its official initial state:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  python scripts/benchmark_client.py --benchmark libero-plus --benchmark-path /path/to/LIBERO-plus \
      --suite "$suite" --episodes 1 || break
done
```

| Flag | Meaning |
|---|---|
| `--suite` | Perturbed task suite: `libero_spatial` (default), `libero_object`, `libero_goal`, or `libero_10` |
| `--task-id` | Perturbed task index in the suite; omit to run all tasks |
| `--libero-config` | Directory containing `config.yaml`; configured automatically if omitted |

The four suites contain 2,402 / 2,518 / 2,591 / 2,519 tasks respectively:
**10,030 episodes** in total. Keep `--episodes 1 --episode-start 0`.
Step limits match LIBERO. See the [official evaluation instructions](https://github.com/sylvestf/LIBERO-plus#-evaluation)
and [suite definitions](https://github.com/sylvestf/LIBERO-plus/blob/main/libero/libero/benchmark/__init__.py).

## SimplerEnv (WidowX)

Serve a Bridge/WidowX model. Run **120 episodes per task**, **480 episodes** in total:

```bash
for task in widowx_spoon_on_towel widowx_carrot_on_plate \
            widowx_stack_cube widowx_put_eggplant_in_basket; do
  python scripts/benchmark_client.py --benchmark simplerenv --benchmark-path /path/to/SimplerEnv \
      --task "$task" --episodes 120 --seed 1000 || break
done
```

| Flag | Meaning |
|---|---|
| `--task` | `widowx_spoon_on_towel`, `widowx_carrot_on_plate`, `widowx_stack_cube` (default), or `widowx_put_eggplant_in_basket` |

`--task` selects one task per command. The loop above runs all four tasks, 120 episodes each.

Episodes use seeds `seed + episode_start` onward. Default step limit: 240.
This client supports one-camera, vision-only Bridge/WidowX policies at 5 Hz.
Task names follow the [official registry](https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/__init__.py).

## Evaluating faster

Run multiple simulator clients against one server; the server batches their predictions.
The Bash examples below cover the same episodes as the full runs above and save one log per client.
`xargs -P` limits concurrency and returns a nonzero status if a client fails.

**LIBERO: 20 concurrent clients.** Split each task's 50 initial states into two disjoint groups of 25.

```bash
mkdir -p outputs/eval/libero
for suite in libero_spatial libero_object libero_goal libero_10; do
  for task_id in {0..9}; do
    for start in 0 25; do
      printf '%s %s %s\n' "$suite" "$task_id" "$start"
    done
  done
done | xargs -n 3 -P 20 bash -c '
  python scripts/benchmark_client.py --benchmark libero --benchmark-path /path/to/LIBERO \
      --suite "$1" --task-id "$2" --episode-start "$3" --episodes 25 \
      > "outputs/eval/libero/${1}_${2}_${3}.log" 2>&1
' _
```

**LIBERO-Plus: four concurrent clients**, one per suite; each evaluates all tasks in that suite.

```bash
mkdir -p outputs/eval/libero-plus
printf '%s\n' libero_spatial libero_object libero_goal libero_10 \
  | xargs -n 1 -P 4 bash -c '
    python scripts/benchmark_client.py --benchmark libero-plus --benchmark-path /path/to/LIBERO-plus \
        --suite "$1" --episodes 1 > "outputs/eval/libero-plus/${1}.log" 2>&1
  ' _
```

**SimplerEnv: 20 concurrent clients.** Split each task into six disjoint groups of 20 episodes.

```bash
mkdir -p outputs/eval/simplerenv
for task in widowx_spoon_on_towel widowx_carrot_on_plate \
            widowx_stack_cube widowx_put_eggplant_in_basket; do
  for start in 0 20 40 60 80 100; do
    printf '%s %s\n' "$task" "$start"
  done
done | xargs -n 2 -P 20 bash -c '
  python scripts/benchmark_client.py --benchmark simplerenv --benchmark-path /path/to/SimplerEnv \
      --task "$1" --seed 1000 --episode-start "$2" --episodes 20 \
      > "outputs/eval/simplerenv/${1}_${2}.log" 2>&1
' _
```

Combine logs by summing `successes` and `episodes` from each client's final JSON summary;
the overall success rate is `total_successes / total_episodes`.

Use `python scripts/benchmark_client.py --benchmark <name> --help` for that benchmark's flags.
For custom clients, see the [WebSocket protocol](protocol.md).
