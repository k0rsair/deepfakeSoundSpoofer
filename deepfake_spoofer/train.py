from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepfake_spoofer.data import ASVspoof5Dataset, collate_audio, make_balanced_sampler
from deepfake_spoofer.model import build_spoof_detector


class TrainableEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        one_minus_decay = 1.0 - self.decay
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=one_minus_decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = parameter.detach().clone()
            parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wav2vec + PyAra/AASIST-style deepfake audio detector.")
    parser.add_argument("--data-dir", default="data", help="Folder with ASVspoof5 TSV files and flac_* folders.")
    parser.add_argument("--output-dir", default="runs/wav2vec_pyara", help="Where checkpoints and logs are written.")
    parser.add_argument(
        "--model-type",
        default="fusion",
        choices=["fusion", "wav2vec_pyara", "mfcc_resnet"],
        help="fusion joins wav2vec/PyAra and MFCC/ResNet embeddings.",
    )
    parser.add_argument("--bundle", default="WAV2VEC2_XLSR_300M", help="torchaudio wav2vec bundle name.")
    parser.add_argument(
        "--wav2vec-cache-dir",
        default=None,
        help="Optional path to precomputed frozen wav2vec features for the selected layer.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-wav2vec", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--wav2vec-layers", type=int, default=None, help="Use only the first N transformer layers.")
    parser.add_argument("--freeze-wav2vec", action="store_true", help="Disable wav2vec fine-tuning.")
    parser.add_argument("--unfreeze-feature-extractor", action="store_true", help="Also train wav2vec conv extractor.")
    parser.add_argument("--freeze-transformer-layers", type=int, default=0, help="Freeze first N transformer layers.")
    parser.add_argument("--no-balanced-sampler", action="store_true", help="Disable class-balanced sampling.")
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-frequency class weights in CE loss.")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--eval-passes", type=int, default=3, help="Average this many validation passes.")
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=3)
    parser.add_argument("--best-metric", default="roc_auc", choices=["roc_auc", "loss"])
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace) -> tuple[ASVspoof5Dataset, ASVspoof5Dataset, DataLoader, DataLoader]:
    ssl_cache_dir = None
    if args.freeze_wav2vec and args.wav2vec_cache_dir and args.model_type in {"fusion", "wav2vec_pyara"}:
        ssl_cache_dir = args.wav2vec_cache_dir

    train_dataset = ASVspoof5Dataset(
        args.data_dir,
        "train",
        sample_rate=16_000,
        max_seconds=args.max_seconds,
        random_crop=True,
        limit=args.limit_train,
        limit_per_class=args.limit_per_class,
        ssl_cache_dir=ssl_cache_dir,
    )
    dev_dataset = ASVspoof5Dataset(
        args.data_dir,
        "dev",
        sample_rate=16_000,
        max_seconds=args.max_seconds,
        random_crop=False,
        limit=args.limit_dev,
        limit_per_class=args.limit_per_class,
        ssl_cache_dir=ssl_cache_dir,
    )

    sampler = None if args.no_balanced_sampler else make_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_audio,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_audio,
    )
    return train_dataset, dev_dataset, train_loader, dev_loader


def validate_wav2vec_cache(args: argparse.Namespace) -> None:
    if not (args.freeze_wav2vec and args.wav2vec_cache_dir and args.model_type in {"fusion", "wav2vec_pyara"}):
        return

    meta_path = Path(args.wav2vec_cache_dir) / "meta.json"
    if not meta_path.exists():
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cached_bundle = meta.get("bundle")
    cached_layers = meta.get("wav2vec_layers")
    if cached_bundle and cached_bundle != args.bundle:
        raise ValueError(
            f"wav2vec cache bundle mismatch: cache has {cached_bundle}, train requested {args.bundle}"
        )
    if cached_layers is not None and cached_layers != args.wav2vec_layers:
        raise ValueError(
            f"wav2vec cache layer mismatch: cache has layer {cached_layers}, train requested {args.wav2vec_layers}"
        )


def make_model(args: argparse.Namespace) -> nn.Module:
    return build_spoof_detector(
        model_type=args.model_type,
        bundle_name=args.bundle,
        freeze_wav2vec=args.freeze_wav2vec,
        freeze_feature_extractor=not args.unfreeze_feature_extractor,
        freeze_transformer_layers=args.freeze_transformer_layers,
        wav2vec_layers=args.wav2vec_layers,
    )


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    wav2vec_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("frontend.wav2vec") or ".frontend.wav2vec" in name:
            wav2vec_params.append(parameter)
        else:
            head_params.append(parameter)

    groups: list[dict[str, Any]] = []
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head})
    if wav2vec_params:
        groups.append({"params": wav2vec_params, "lr": args.lr_wav2vec})
    if not groups:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if args.scheduler == "none":
        return None

    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = min(int(total_steps * args.warmup_ratio), max(total_steps - 1, 0))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr_scale + (1.0 - args.min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def class_weight_tensor(dataset: ASVspoof5Dataset, device: torch.device) -> torch.Tensor:
    counts = dataset.class_counts
    total = sum(counts.values())
    weights = [total / max(counts.get(class_id, 1), 1) for class_id in range(2)]
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights / weights.mean()


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    ssl_features = batch["ssl_features"]
    ssl_lengths = batch["ssl_lengths"]
    return {
        "waveforms": batch["waveforms"].to(device, non_blocking=True),
        "lengths": batch["lengths"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "ssl_features": ssl_features.to(device, non_blocking=True) if ssl_features is not None else None,
        "ssl_lengths": ssl_lengths.to(device, non_blocking=True) if ssl_lengths is not None else None,
    }


def compute_metrics(labels: list[int], logits: list[list[float]]) -> dict[str, float]:
    y_true = np.asarray(labels)
    raw_logits = np.asarray(logits)
    y_pred = raw_logits.argmax(axis=1)
    exp_logits = np.exp(raw_logits - raw_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    metrics = {"accuracy": float((y_pred == y_true).mean())}
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    else:
        metrics["roc_auc"] = math.nan
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
    eval_passes: int = 1,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels: np.ndarray | None = None
    logits_sum: np.ndarray | None = None

    dataset = getattr(loader, "dataset", None)
    original_random_crop = getattr(dataset, "random_crop", False) if dataset is not None else False

    if dataset is not None and hasattr(dataset, "random_crop"):
        dataset.random_crop = eval_passes > 1

    try:
        for eval_pass in range(eval_passes):
            pass_loss = 0.0
            pass_items = 0
            pass_logits: list[list[float]] = []
            pass_labels: list[int] = []

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"dev[{eval_pass + 1}/{eval_passes}]", leave=False):
                    batch_on_device = batch_to_device(batch, device)
                    waveforms = batch_on_device["waveforms"]
                    lengths = batch_on_device["lengths"]
                    targets = batch_on_device["labels"]
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        _, logits = model(
                            waveforms,
                            lengths,
                            ssl_features=batch_on_device["ssl_features"],
                            ssl_feature_lengths=batch_on_device["ssl_lengths"],
                        )
                        loss = criterion(logits, targets)
                    batch_size = targets.size(0)
                    pass_loss += float(loss.detach().cpu()) * batch_size
                    pass_items += batch_size
                    pass_labels.extend(targets.detach().cpu().tolist())
                    pass_logits.extend(logits.detach().float().cpu().tolist())

            current_logits = np.asarray(pass_logits)
            current_labels = np.asarray(pass_labels)
            total_loss += pass_loss / max(pass_items, 1)
            if logits_sum is None:
                logits_sum = current_logits
                all_labels = current_labels
            else:
                logits_sum += current_logits
    finally:
        if dataset is not None and hasattr(dataset, "random_crop"):
            dataset.random_crop = original_random_crop

    averaged_logits = logits_sum / max(eval_passes, 1)
    metrics = compute_metrics(all_labels.tolist(), averaged_logits.tolist())
    metrics["loss"] = total_loss / max(eval_passes, 1)
    return metrics


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric_value: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric_value": best_metric_value,
            "config": vars(args),
        },
        path,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    ema: TrainableEMA | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_items = 0
    labels: list[int] = []
    logits_rows: list[list[float]] = []
    use_amp = args.mixed_precision and device.type == "cuda"

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        batch_on_device = batch_to_device(batch, device)
        waveforms = batch_on_device["waveforms"]
        lengths = batch_on_device["lengths"]
        targets = batch_on_device["labels"]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _, logits = model(
                waveforms,
                lengths,
                ssl_features=batch_on_device["ssl_features"],
                ssl_feature_lengths=batch_on_device["ssl_lengths"],
            )
            loss = criterion(logits, targets) / args.grad_accum_steps

        scaler.scale(loss).backward()

        if step % args.grad_accum_steps == 0:
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_ran = (not use_amp) or (scaler.get_scale() >= previous_scale)
            if scheduler is not None and optimizer_ran:
                scheduler.step()
            if ema is not None and optimizer_ran:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * args.grad_accum_steps * batch_size
        total_items += batch_size
        labels.extend(targets.detach().cpu().tolist())
        logits_rows.extend(logits.detach().float().cpu().tolist())

    if len(loader) % args.grad_accum_steps != 0:
        if args.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_ran = (not use_amp) or (scaler.get_scale() >= previous_scale)
        if scheduler is not None and optimizer_ran:
            scheduler.step()
        if ema is not None and optimizer_ran:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    metrics = compute_metrics(labels, logits_rows)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    validate_wav2vec_cache(args)
    train_dataset, dev_dataset, train_loader, dev_loader = make_loaders(args)
    print(f"train items: {len(train_dataset)} class_counts={dict(train_dataset.class_counts)}")
    print(f"dev items: {len(dev_dataset)} class_counts={dict(dev_dataset.class_counts)}")
    print(f"device: {device}")
    print(f"wav2vec cache: {args.wav2vec_cache_dir if train_dataset.ssl_cache_dir is not None else 'disabled'}")

    model = make_model(args).to(device)
    optimizer = make_optimizer(model, args)
    optimizer_steps_per_epoch = max(math.ceil(len(train_loader) / max(args.grad_accum_steps, 1)), 1)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=optimizer_steps_per_epoch)
    class_weights = class_weight_tensor(train_dataset, device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    use_amp = args.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = TrainableEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    best_metric_value = -math.inf if args.best_metric == "roc_auc" else math.inf
    epochs_without_improve = 0

    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                ema,
                device,
                args,
            )
            if ema is not None:
                ema.apply_shadow(model)
            dev_metrics = evaluate(
                model,
                dev_loader,
                criterion,
                device,
                use_amp=use_amp,
                eval_passes=max(args.eval_passes, 1),
            )
            if ema is not None:
                ema.restore(model)

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "dev": dev_metrics,
                "lr": [group["lr"] for group in optimizer.param_groups],
            }
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics_file.flush()

            print(
                f"epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"dev_loss={dev_metrics['loss']:.4f} dev_acc={dev_metrics['accuracy']:.4f} "
                f"dev_auc={dev_metrics['roc_auc']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            current_metric_value = dev_metrics[args.best_metric]
            if args.best_metric == "loss":
                improved = current_metric_value < best_metric_value
            else:
                improved = not math.isnan(current_metric_value) and current_metric_value > best_metric_value

            if improved:
                best_metric_value = current_metric_value
                epochs_without_improve = 0
                if ema is not None:
                    ema.apply_shadow(model)
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric_value=best_metric_value,
                    args=args,
                )
                if ema is not None:
                    ema.restore(model)
            else:
                epochs_without_improve += 1

            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric_value=best_metric_value,
                args=args,
            )

            if epoch >= args.early_stopping_min_epochs and epochs_without_improve >= args.early_stopping_patience:
                print(
                    f"early stopping at epoch {epoch}: "
                    f"no improvement in {args.best_metric} for {epochs_without_improve} epochs"
                )
                break


if __name__ == "__main__":
    main()
