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
from deepfake_spoofer.model import Wav2VecPyAraSpoofDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wav2vec + PyAra/AASIST-style deepfake audio detector.")
    parser.add_argument("--data-dir", default="data", help="Folder with ASVspoof5 TSV files and flac_* folders.")
    parser.add_argument("--output-dir", default="runs/wav2vec_pyara", help="Where checkpoints and logs are written.")
    parser.add_argument("--bundle", default="WAV2VEC2_XLSR_300M", help="torchaudio wav2vec bundle name.")
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
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace) -> tuple[ASVspoof5Dataset, ASVspoof5Dataset, DataLoader, DataLoader]:
    train_dataset = ASVspoof5Dataset(
        args.data_dir,
        "train",
        sample_rate=16_000,
        max_seconds=args.max_seconds,
        random_crop=True,
        limit=args.limit_train,
        limit_per_class=args.limit_per_class,
    )
    dev_dataset = ASVspoof5Dataset(
        args.data_dir,
        "dev",
        sample_rate=16_000,
        max_seconds=args.max_seconds,
        random_crop=False,
        limit=args.limit_dev,
        limit_per_class=args.limit_per_class,
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


def make_model(args: argparse.Namespace) -> Wav2VecPyAraSpoofDetector:
    return Wav2VecPyAraSpoofDetector(
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
        if name.startswith("frontend.wav2vec"):
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


def class_weight_tensor(dataset: ASVspoof5Dataset, device: torch.device) -> torch.Tensor:
    counts = dataset.class_counts
    total = sum(counts.values())
    weights = [total / max(counts.get(class_id, 1), 1) for class_id in range(2)]
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights / weights.mean()


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["waveforms"].to(device, non_blocking=True),
        batch["lengths"].to(device, non_blocking=True),
        batch["labels"].to(device, non_blocking=True),
    )


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
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    labels: list[int] = []
    logits_rows: list[list[float]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="dev", leave=False):
            waveforms, lengths, targets = batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, logits = model(waveforms, lengths)
                loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
            labels.extend(targets.detach().cpu().tolist())
            logits_rows.extend(logits.detach().float().cpu().tolist())

    metrics = compute_metrics(labels, logits_rows)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_auc: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_auc": best_auc,
            "config": vars(args),
        },
        path,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
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
        waveforms, lengths, targets = batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _, logits = model(waveforms, lengths)
            loss = criterion(logits, targets) / args.grad_accum_steps

        scaler.scale(loss).backward()

        if step % args.grad_accum_steps == 0:
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
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
        scaler.step(optimizer)
        scaler.update()
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

    train_dataset, dev_dataset, train_loader, dev_loader = make_loaders(args)
    print(f"train items: {len(train_dataset)} class_counts={dict(train_dataset.class_counts)}")
    print(f"dev items: {len(dev_dataset)} class_counts={dict(dev_dataset.class_counts)}")
    print(f"device: {device}")

    model = make_model(args).to(device)
    optimizer = make_optimizer(model, args)
    class_weights = class_weight_tensor(train_dataset, device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = args.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_auc = -math.inf
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args)
            dev_metrics = evaluate(model, dev_loader, criterion, device, use_amp=use_amp)

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "dev": dev_metrics,
            }
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics_file.flush()

            print(
                f"epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"dev_loss={dev_metrics['loss']:.4f} dev_acc={dev_metrics['accuracy']:.4f} "
                f"dev_auc={dev_metrics['roc_auc']:.4f}"
            )

            current_auc = dev_metrics["roc_auc"]
            if not math.isnan(current_auc) and current_auc > best_auc:
                best_auc = current_auc
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_auc=best_auc,
                    args=args,
                )

            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_auc=best_auc,
                args=args,
            )


if __name__ == "__main__":
    main()
