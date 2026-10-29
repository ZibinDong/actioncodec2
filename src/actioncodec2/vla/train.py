"""Train G0.5 with one of the released action tokenizers."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from actioncodec2.vla.data import BalancedReplacementSampler
from actioncodec2.vla.runtime import (
    DTYPES,
    build_dataset,
    build_model,
    build_processor,
    build_tokenizer_for,
    save_run,
)
from actioncodec2.vla.training.config import RunConfig
from actioncodec2.vla.training.distributed import Distributed, cleanup, setup
from actioncodec2.vla.training.optim import parameter_groups
from actioncodec2.vla.training.tracking import BACKENDS, RunTracker

__all__ = ["main", "train"]


#: ``static_graph=True`` is deliberately absent: combined with ``no_sync`` gradient
#: accumulation it produced wrong, rank-dependent gradients (see
#: ``tests/test_training.py``).  Every parameter of the model receives a
#: gradient on every step, so unused-parameter detection is not needed.
DDP_OPTIONS = {"gradient_as_bucket_view": True}


def _compute_context(config: RunConfig, device: str):
    """Autocast to ``model.dtype`` on CUDA over FP32 parameters; a no-op otherwise."""
    dtype = DTYPES[config.model.dtype]
    if torch.device(device).type == "cuda" and dtype != torch.float32:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _training_sampler(dataset, config: RunConfig, context: Distributed):
    if config.data.sample_order == "random":
        if context.enabled:
            return DistributedSampler(
                dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=config.train.seed,
                drop_last=False,
            )
        return RandomSampler(dataset, generator=torch.Generator().manual_seed(config.train.seed))
    # With replacement a pass has no fixed membership; give it the per-rank
    # length of a shuffled pass so step counts stay comparable.
    return BalancedReplacementSampler(
        dataset.entry_roots(),
        dataset.offsets,
        num_samples=max(1, len(dataset) // context.world_size),
        seed=config.train.seed,
        rank=context.rank,
        world_size=context.world_size,
        root_weights=config.data.root_weights,
    )


def _loader_options(train_config) -> dict[str, object]:
    workers = int(train_config.num_workers)
    return {
        "num_workers": workers,
        "persistent_workers": workers > 0,
        # PyTorch requires timeout=0 without worker processes.
        "timeout": float(train_config.dataloader_timeout_seconds) if workers > 0 else 0,
    }


def _next_batch(batches: Iterator, *, config: RunConfig, rank: int):
    """Fetch one batch, reporting a DataLoader timeout as a stalled video worker."""
    try:
        return next(batches)
    except RuntimeError as error:
        if "DataLoader timed out" not in str(error):
            raise
        raise RuntimeError(
            f"rank {rank}: no batch arrived within "
            f"{config.train.dataloader_timeout_seconds:g} seconds (stalled video decoding?)"
        ) from error


def _validate_optimizer_fp32(optimizer: torch.optim.Optimizer) -> None:
    """Fail if any parameter, gradient or Adam moment is not FP32."""
    for parameter, state in optimizer.state.items():
        if parameter.dtype != torch.float32:
            raise TypeError(f"optimizer parameter must be FP32, got {parameter.dtype}")
        if parameter.grad is not None and parameter.grad.dtype != torch.float32:
            raise TypeError(f"optimizer gradient must be FP32, got {parameter.grad.dtype}")
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if value.dtype != torch.float32:
                    raise TypeError(f"optimizer state {name} must be FP32, got {value.dtype}")


def _schedule(
    step: int, *, total: int, warmup: int, min_ratio: float, decay_fraction: float = 1.0
) -> float:
    """LR multiplier: linear warmup, a plateau, then cosine decay to ``min_ratio``.

    ``decay_fraction`` is the share of the post-warmup steps spent decaying;
    ``1.0`` starts the cosine right after warmup.
    """
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    tail = min(max(decay_fraction, 1e-9), 1.0)
    if progress < 1.0 - tail:
        return 1.0
    decayed = (progress - (1.0 - tail)) / tail
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(decayed, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


def _ddp_loss_scale(local_tokens: int, global_tokens: int | float, world_size: int) -> float:
    """Scale a rank's token-mean loss so DDP's gradient average is the global token mean."""
    if local_tokens <= 0 or global_tokens <= 0 or world_size < 1:
        raise ValueError("loss token counts and world size must be positive")
    return float(world_size * local_tokens / global_tokens)


def _correct_tokens(outputs, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Supervised tokens predicted correctly, as a device scalar (no host sync)."""
    mask = batch["action_label_mask"].bool()
    predictions = outputs.logits.detach().argmax(dim=-1)
    return predictions[mask].eq(batch["action_labels"][mask]).sum()


def _accumulate_gradients(
    parallel,
    host_batches: list[dict[str, torch.Tensor]],
    token_counts: list[int],
    *,
    global_tokens: float,
    context: Distributed,
    device: str,
    compute: Callable[[], object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backpropagate every microbatch of one optimizer step."""
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    last = len(host_batches) - 1
    for index, (batch, tokens) in enumerate(zip(host_batches, token_counts, strict=True)):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        defer_sync = context.enabled and index < last
        with parallel.no_sync() if defer_sync else nullcontext():
            with compute():
                outputs = parallel(**batch)
            scale = _ddp_loss_scale(tokens, global_tokens, context.world_size)
            (outputs.loss * scale).backward()
        loss_sum += outputs.loss.detach().float() * tokens
        correct += _correct_tokens(outputs, batch)
    return loss_sum, correct


def _make_scheduler(optimizer, config: RunConfig, *, resume_step: int):
    train = config.train
    total = train.schedule_steps if train.schedule_steps is not None else train.steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _schedule(
            step,
            total=total,
            warmup=train.warmup_steps,
            min_ratio=train.min_lr_ratio,
            decay_fraction=train.decay_fraction,
        ),
    )
    if resume_step:
        # Continue the reference curve: the first resumed step uses the rate
        # step ``resume_step + 1`` of the original run would have used.
        scheduler.last_epoch = resume_step
        scheduler._step_count = resume_step + 1
        scheduler._last_lr = []
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs, strict=True):
            group["lr"] = base_lr * scheduler.lr_lambdas[0](resume_step)
            scheduler._last_lr.append(group["lr"])
    return scheduler, total


def _optimize(
    *,
    model,
    batches: Iterator,
    loader: DataLoader,
    config: RunConfig,
    device: str,
    history: list[dict[str, float]],
    started: float,
    tracker: RunTracker,
    context: Distributed,
    resume_step: int,
    save: Callable[[int, torch.optim.Optimizer], None],
    start_epoch: int = 0,
) -> None:
    """Run optimizer steps ``resume_step + 1 .. train.steps``."""
    optimizer = torch.optim.AdamW(
        parameter_groups(model, weight_decay=config.train.weight_decay),
        lr=config.train.learning_rate,
        betas=(0.9, 0.95),
    )
    if resume_step and config.train.warm_start_checkpoint:
        if _restore_optimizer_state(optimizer, Path(config.train.warm_start_checkpoint)):
            if context.is_primary:
                print("[warm-start] optimizer state restored", flush=True)
        elif context.is_primary:
            print("[warm-start] no optimizer state in the checkpoint; moments restart", flush=True)
    scheduler, horizon = _make_scheduler(optimizer, config, resume_step=resume_step)
    parallel = model
    if context.enabled:
        parallel = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if torch.device(device).type == "cuda" else None,
            **DDP_OPTIONS,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if context.is_primary:
        print(
            f"[train] steps {resume_step + 1}..{config.train.steps} | "
            f"lr {config.train.learning_rate:.2e} | schedule horizon {horizon} | "
            f"{sum(p.numel() for p in trainable) / 1e6:.1f}M trainable parameters",
            flush=True,
        )

    epoch = start_epoch
    rank = context.rank
    for step in range(resume_step + 1, config.train.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        host_batches = []
        for _ in range(config.train.gradient_accumulation):
            try:
                batch = _next_batch(batches, config=config, rank=rank)
            except StopIteration:
                epoch += 1
                sampler = getattr(loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                batches = iter(loader)
                batch = _next_batch(batches, config=config, rank=rank)
            host_batches.append(batch)

        # The model returns the mean over this rank's supervised tokens, and
        # token counts differ across ranks and microbatches.  Rescaling each
        # microbatch by its share of the global count makes the DDP-averaged
        # gradient the gradient of the global token mean.
        token_counts = [int(batch["action_label_mask"].sum()) for batch in host_batches]
        local_tokens = sum(token_counts)
        if local_tokens <= 0:
            raise RuntimeError("a training step contains no supervised tokens")
        global_tokens = context.sum_(local_tokens)[0]

        loss_sum, correct = _accumulate_gradients(
            parallel,
            host_batches,
            token_counts,
            global_tokens=global_tokens,
            context=context,
            device=device,
            compute=lambda: _compute_context(config, device),
        )

        norm = torch.nn.utils.clip_grad_norm_(
            trainable, config.train.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        if step == resume_step + 1:
            _validate_optimizer_fp32(optimizer)
            # The first batches were checked for range, padding and finiteness; the
            # per-step host synchronizations are not worth paying again.
            model.validate_inputs = False
        scheduler.step()

        if (config.train.save_every and step % config.train.save_every == 0) or (
            step in config.train.save_steps
        ):
            save(step, optimizer)

        if step % config.train.log_every == 0 or step == resume_step + 1:
            summed_loss, summed_tokens, summed_correct, summed_counted = context.sum_(
                float(loss_sum), local_tokens, float(correct), local_tokens
            )
            metrics = {
                "train/loss": summed_loss / max(summed_tokens, 1.0),
                "train/token_accuracy": summed_correct / max(summed_counted, 1.0),
                "train/grad_norm": float(norm),
                "train/lr": scheduler.get_last_lr()[0],
            }
            tracker.log(metrics, step=step)
            record = {
                "step": step,
                "seconds": time.time() - started,
                **{key.removeprefix("train/"): value for key, value in metrics.items()},
            }
            history.append(record)
            if context.is_primary:
                print(
                    f"[train] step {step:>6} loss {record['loss']:.4f} "
                    f"acc {record['token_accuracy']:.4f} grad {record['grad_norm']:.2f} "
                    f"lr {record['lr']:.2e} {record['seconds']:.0f}s",
                    flush=True,
                )


OPTIMIZER_STATE_FILE = "optimizer.pt"


def _save_optimizer_state(directory: Path, optimizer: torch.optim.Optimizer) -> None:
    torch.save(optimizer.state_dict(), directory / OPTIMIZER_STATE_FILE)


def _restore_optimizer_state(optimizer: torch.optim.Optimizer, checkpoint: Path) -> bool:
    """Load ``optimizer.pt`` from ``checkpoint`` if it has one; whether it did."""
    path = checkpoint / OPTIMIZER_STATE_FILE
    if not path.is_file():
        return False
    optimizer.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return True


def _warm_start_model(model, checkpoint: Path) -> None:
    """Load every parameter from a weights-only ``save_run`` checkpoint."""
    if not (checkpoint / "_SUCCESS").is_file():
        raise ValueError(f"warm-start checkpoint has no _SUCCESS marker: {checkpoint}")
    shards = sorted(checkpoint.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"warm-start checkpoint has no model weights: {checkpoint}")
    from safetensors import safe_open

    # Copy tensor by tensor from memory-mapped shards so no rank holds a second
    # full copy of the weights in host memory.
    target = model.state_dict()
    loaded: set[str] = set()
    unexpected: list[str] = []
    with torch.no_grad():
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key not in target:
                        unexpected.append(key)
                        continue
                    value = handle.get_tensor(key)
                    if value.shape != target[key].shape:
                        raise ValueError(
                            f"warm-start tensor {key} has shape {tuple(value.shape)}, "
                            f"expected {tuple(target[key].shape)}"
                        )
                    target[key].copy_(value)
                    loaded.add(key)
    missing = sorted(set(target) - loaded)
    if unexpected or missing:
        raise ValueError(
            f"warm-start checkpoint does not match the model: "
            f"unexpected={sorted(unexpected)[:4]}, missing={missing[:4]}"
        )


def train(config: RunConfig, *, device: str = "cuda") -> dict[str, object]:
    """Train a G0.5 policy and return a report of the run."""
    config.validate_for_training()
    context = setup()
    if context.enabled:
        device = context.device
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)

    tokenizer = build_tokenizer_for(config)
    processor = build_processor(config, tokenizer)
    dataset = build_dataset(config, processor, tokenizer)
    model = build_model(config, processor, device=device)
    non_fp32 = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.dtype != torch.float32
    ]
    if non_fp32:
        raise TypeError(f"training requires FP32 master parameters, got {non_fp32[:4]}")

    resume_step = 0
    start_epoch = 0
    if config.train.warm_start_checkpoint:
        _warm_start_model(model, Path(config.train.warm_start_checkpoint))
        resume_step = int(config.train.warm_start_step)
        if context.is_primary:
            print(
                f"[warm-start] weights from {config.train.warm_start_checkpoint} "
                f"(global step {resume_step})",
                flush=True,
            )

    sampler = _training_sampler(dataset, config, context)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        sampler=sampler,
        collate_fn=processor.collator(),
        drop_last=False,
        **_loader_options(config.train),
    )
    global_batch = config.train.batch_size * context.world_size * config.train.gradient_accumulation
    if resume_step:
        # Resume the data order at the epoch the checkpoint falls in; the exact
        # position inside the epoch is not recoverable from weights.
        start_epoch = resume_step * global_batch // max(len(dataset), 1)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(start_epoch)

    output = Path(config.train.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    model.train()
    started = time.time()
    batches = iter(loader)
    if context.is_primary:
        print(
            f"training on {len(dataset)} samples from {len(dataset.entries)} episodes | "
            f"tokenizer {tokenizer.name} (vocab {tokenizer.vocab_size}, previous action "
            f"{'required' if tokenizer.requires_previous_action else 'not needed'}) | "
            f"world size {context.world_size} | global batch {global_batch}",
            flush=True,
        )

    tracker = RunTracker.start(
        config.tracking if context.is_primary else None,
        run_config={
            **{key: value for key, value in config.to_json().items() if key != "tracking"},
            "run": {
                "samples": len(dataset),
                "episodes": len(dataset.entries),
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "requires_previous_action": tokenizer.requires_previous_action,
                "execution_stride": dataset.execution_stride,
                "world_size": context.world_size,
                "global_batch_size": global_batch,
            },
        },
        name=output.name,
    )

    def persist(directory: Path) -> None:
        if context.is_primary:
            save_run(directory, model, processor, config, execution_stride=dataset.execution_stride)
        # No rank continues until the checkpoint is complete.
        context.barrier()

    def checkpoint(step: int, optimizer: torch.optim.Optimizer) -> None:
        directory = output / f"step-{step:06d}"
        persist(directory)
        if context.is_primary:
            if config.train.save_optimizer_state:
                _save_optimizer_state(directory, optimizer)
            # Written last, so watchers never read a half-written checkpoint.
            (directory / "_SUCCESS").write_text("ready\n", encoding="utf-8")
            print(f"[checkpoint] {directory}", flush=True)

    try:
        _optimize(
            model=model,
            batches=batches,
            loader=loader,
            config=config,
            device=device,
            history=history,
            started=started,
            tracker=tracker,
            context=context,
            resume_step=resume_step,
            save=checkpoint,
            start_epoch=start_epoch,
        )
        if config.train.save_final:
            persist(output)
        minutes = (time.time() - started) / 60.0
        final = history[-1] if history else {}
        tracker.summarize(
            {
                "loss": final.get("loss"),
                "token_accuracy": final.get("token_accuracy"),
                "minutes": round(minutes, 2),
            }
        )
        report = {
            "steps": config.train.steps,
            "samples": len(dataset),
            "episodes": len(dataset.entries),
            "tokenizer": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "requires_previous_action": tokenizer.requires_previous_action,
            "minutes": minutes,
            "final": final,
            "history": history,
        }
        if context.is_primary:
            (output / "train_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    except BaseException as error:
        state = "aborted" if isinstance(error, KeyboardInterrupt) else "crashed"
        tracker.finish(state=state, error=f"{type(error).__name__}: {error}")
        cleanup()
        raise
    tracker.finish(state="success")
    cleanup()
    return report


def _latest_complete_checkpoint(output: Path) -> tuple[int, Path] | None:
    """Newest ``step-NNNNNN`` directory whose ``_SUCCESS`` marker exists."""
    candidates: list[tuple[int, Path]] = []
    for directory in output.glob("step-*"):
        digits = directory.name.partition("step-")[2]
        if directory.is_dir() and digits.isdigit() and (directory / "_SUCCESS").is_file():
            candidates.append((int(digits), directory))
    return max(candidates, default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a G0.5 action policy")
    parser.add_argument("--config", type=Path, required=True, help="recipe under configs/")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="directory holding the LeRobot datasets the recipe lists in data.roots",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        help="G0.5 base model_state_dict.pt (default: download it from OpenGalaxea/G05)",
    )
    parser.add_argument(
        "--text-tokenizer",
        help="G0.5 Qwen3.5 text tokenizer directory (default: the copy in OpenGalaxea/G05)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="override train.steps")
    parser.add_argument("--output-dir", default=None, help="override train.output_dir")
    parser.add_argument("--learning-rate", type=float, default=None, help="override train.learning_rate")
    parser.add_argument("--save-every", type=int, default=None, help="override train.save_every")
    parser.add_argument(
        "--no-token-cache",
        action="store_true",
        help="encode action chunks on demand instead of building the token cache",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="write no periodic or final checkpoints"
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="continue from the newest complete step-* checkpoint in the output directory: "
        "weights and the LR schedule resume, and so does the optimizer state when the run "
        "saved it (train.save_optimizer_state); otherwise Adam moments restart from zero",
    )
    parser.add_argument("--tracker", choices=BACKENDS, default=None, help="override tracking.backend")
    parser.add_argument("--run-name", default=None, help="override tracking.name")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved run configuration and exit",
    )
    args = parser.parse_args(argv)

    try:
        config = RunConfig.load(
            args.config,
            data_root=args.data_root,
            base_checkpoint=args.base_checkpoint,
            text_tokenizer=args.text_tokenizer,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.steps is not None:
        config.train.steps = args.steps
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    if args.save_every is not None:
        config.train.save_every = args.save_every
    if args.no_token_cache:
        config.data.token_cache = None
    if args.no_save:
        config.train.save_every = 0
        config.train.save_steps = ()
        config.train.save_final = False
    if args.tracker is not None:
        config.tracking.backend = args.tracker
    if args.run_name is not None:
        config.tracking.name = args.run_name
    if args.print_config:
        print(json.dumps(config.to_json(), indent=2))
        return 0

    output = Path(config.train.output_dir)
    finished = any(
        (output / name).exists()
        for name in ("train_report.json", "model.safetensors", "model.safetensors.index.json")
    )
    if args.auto_resume:
        if finished:
            parser.error(f"the run in {output} already finished; auto-resume refuses to continue it")
        latest = _latest_complete_checkpoint(output)
        if latest is None:
            print(f"[auto-resume] no complete checkpoint in {output}; starting fresh", flush=True)
        else:
            resume_step, resume_directory = latest
            if resume_step >= config.train.steps:
                parser.error(
                    f"checkpoint {resume_directory} is at step {resume_step}, at or past "
                    f"the requested {config.train.steps} steps"
                )
            config.train.warm_start_checkpoint = str(resume_directory)
            config.train.warm_start_step = resume_step
            if config.train.schedule_steps is None:
                config.train.schedule_steps = config.train.steps
    elif finished or any(output.glob("step-*")):
        parser.error(
            f"refusing to overwrite an existing training run: {output} "
            "(pass --auto-resume to continue an interrupted one)"
        )
    report = train(config, device=args.device)
    print(json.dumps(report["final"], indent=2))
    return 0
