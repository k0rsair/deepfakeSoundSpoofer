from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepfake_spoofer.data import ASVspoof5Dataset, collate_audio
from deepfake_spoofer.predict import build_model_from_checkpoint
from deepfake_spoofer.train import batch_to_device, class_weight_tensor, needs_waveforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a labeled split.")
    parser.add_argument("--checkpoint", default="runs/wav2vec_pyara/best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "eval"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--eval-passes", type=int, default=None)
    parser.add_argument(
        "--wav2vec-cache-dir",
        default=None,
        help="Override wav2vec cache path from the checkpoint config when evaluating frozen wav2vec runs.",
    )
    parser.add_argument(
        "--mfcc-cache-dir",
        default=None,
        help="Override MFCC cache path from the checkpoint config when evaluating MFCC-based runs.",
    )
    parser.add_argument("--output", default=None, help="Optional path to save the JSON summary.")
    parser.add_argument("--predictions-csv", default=None, help="Optional CSV path for per-file predictions.")
    parser.add_argument("--predictions-jsonl", default=None, help="Optional JSONL path for per-file predictions.")
    parser.add_argument("--plots-dir", default=None, help="Optional directory for PNG plots.")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def dataset_max_seconds(checkpoint: dict[str, Any], requested: float | None) -> float:
    if requested is not None:
        return requested
    return float(checkpoint.get("config", {}).get("max_seconds", 4.0))


def resolve_eval_passes(checkpoint: dict[str, Any], requested: int | None) -> int:
    if requested is not None:
        return max(requested, 1)
    return max(int(checkpoint.get("config", {}).get("eval_passes", 1)), 1)


def resolve_use_amp(checkpoint: dict[str, Any], device: torch.device, requested: bool | None) -> bool:
    if device.type != "cuda":
        return False
    if requested is not None:
        return requested
    return bool(checkpoint.get("config", {}).get("mixed_precision", True))


def split_cache_dir(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / split


def resolve_ssl_cache_dir(args: argparse.Namespace, checkpoint: dict[str, Any]) -> Path | None:
    config = checkpoint.get("config", {})
    ssl_cache_dir = args.wav2vec_cache_dir
    if ssl_cache_dir is None and config.get("freeze_wav2vec") and config.get("model_type") in {
        "fusion",
        "fusion_temporal",
        "wav2vec_pyara",
        "wav2vec_temporal",
    }:
        ssl_cache_dir = config.get("wav2vec_cache_dir")

    if not ssl_cache_dir:
        return None

    split_dir = split_cache_dir(ssl_cache_dir, args.split)
    if split_dir.exists():
        return Path(ssl_cache_dir)

    print(f"cache note: no wav2vec cache found for split '{args.split}' in {split_dir}; using wav2vec directly.")
    return None


def resolve_mfcc_cache_dir(args: argparse.Namespace, checkpoint: dict[str, Any]) -> Path | None:
    config = checkpoint.get("config", {})
    mfcc_cache_dir = args.mfcc_cache_dir
    if mfcc_cache_dir is None and config.get("model_type") in {"fusion", "fusion_temporal", "mfcc_resnet"}:
        mfcc_cache_dir = config.get("mfcc_cache_dir")

    if not mfcc_cache_dir:
        return None

    split_dir = split_cache_dir(mfcc_cache_dir, args.split)
    if split_dir.exists():
        return Path(mfcc_cache_dir)

    print(f"cache note: no MFCC cache found for split '{args.split}' in {split_dir}; computing MFCC on the fly.")
    return None


def make_loader(args: argparse.Namespace, checkpoint: dict[str, Any], sample_rate: int) -> DataLoader:
    ssl_cache_dir = resolve_ssl_cache_dir(args, checkpoint)
    mfcc_cache_dir = resolve_mfcc_cache_dir(args, checkpoint)
    load_waveform = needs_waveforms(
        checkpoint.get("config", {}).get("model_type", "wav2vec_pyara"),
        ssl_cache_dir=ssl_cache_dir,
        mfcc_cache_dir=mfcc_cache_dir,
    )
    dataset = ASVspoof5Dataset(
        args.data_dir,
        args.split,
        sample_rate=sample_rate,
        max_seconds=dataset_max_seconds(checkpoint, args.max_seconds),
        random_crop=False,
        limit=args.limit,
        limit_per_class=args.limit_per_class,
        ssl_cache_dir=ssl_cache_dir,
        mfcc_cache_dir=mfcc_cache_dir,
        load_waveform=load_waveform,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_audio,
    )


def make_criterion(checkpoint: dict[str, Any], dataset: ASVspoof5Dataset, device: torch.device) -> nn.Module:
    config = checkpoint.get("config", {})
    class_weights = class_weight_tensor(dataset, device) if config.get("class_weights", False) else None
    label_smoothing = float(config.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def probability_from_logits(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def compute_eer(labels: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) != 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(labels, spoof_scores, pos_label=1)
    fnr = 1.0 - tpr
    best_index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[best_index] + fnr[best_index]) / 2.0)
    threshold = float(thresholds[best_index])
    return eer, threshold


def compute_metrics_from_logits(labels: np.ndarray, logits: np.ndarray, average_loss: float) -> tuple[dict[str, Any], np.ndarray]:
    probs = probability_from_logits(logits)
    pred = probs.argmax(axis=1)
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tp = int(((pred == 1) & (labels == 1)).sum())

    metrics: dict[str, Any] = {
        "accuracy": float((pred == labels).mean()),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "loss": average_loss,
        "confusion_matrix": {
            "labels": ["bonafide", "spoof"],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }

    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
        metrics["average_precision"] = float(average_precision_score(labels, probs[:, 1]))
        eer, eer_threshold = compute_eer(labels, probs[:, 1])
        metrics["eer"] = eer
        metrics["eer_threshold"] = eer_threshold
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
        metrics["eer"] = float("nan")
        metrics["eer_threshold"] = float("nan")

    return metrics, probs


def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
    eval_passes: int = 1,
    desc_prefix: str = "dev",
    collect_prediction_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, dict[str, np.ndarray]]:
    model.eval()
    total_loss = 0.0
    all_labels: np.ndarray | None = None
    logits_sum: np.ndarray | None = None
    all_paths: list[str] = []
    all_file_ids: list[str] = []

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
                for batch in tqdm(loader, desc=f"{desc_prefix}[{eval_pass + 1}/{eval_passes}]", leave=False):
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
                            mfcc_features=batch_on_device["mfcc_features"],
                            mfcc_feature_lengths=batch_on_device["mfcc_lengths"],
                        )
                        loss = criterion(logits, targets)

                    batch_size = targets.size(0)
                    pass_loss += float(loss.detach().cpu()) * batch_size
                    pass_items += batch_size
                    pass_labels.extend(targets.detach().cpu().tolist())
                    pass_logits.extend(logits.detach().float().cpu().tolist())

                    if eval_pass == 0:
                        all_paths.extend(batch["paths"])
                        all_file_ids.extend(batch["file_ids"])

            current_logits = np.asarray(pass_logits, dtype=np.float64)
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
    metrics, probs = compute_metrics_from_logits(all_labels, averaged_logits, total_loss / max(eval_passes, 1))
    pred_ids = probs.argmax(axis=1)
    predictions = None
    if collect_prediction_rows:
        predictions = [
            {
                "file_id": file_id,
                "path": path,
                "label": int(label),
                "pred_label": int(pred_label),
                "p_bonafide": float(prob_row[0]),
                "p_spoof": float(prob_row[1]),
                "logit_bonafide": float(logit_row[0]),
                "logit_spoof": float(logit_row[1]),
            }
            for file_id, path, label, pred_label, prob_row, logit_row in zip(
                all_file_ids,
                all_paths,
                all_labels.tolist(),
                pred_ids.tolist(),
                probs.tolist(),
                averaged_logits.tolist(),
                strict=True,
            )
        ]
    plot_payload = {
        "labels": all_labels,
        "probs": probs,
        "pred_ids": pred_ids,
    }
    return metrics, predictions, plot_payload


def write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_id",
                "path",
                "label",
                "pred_label",
                "p_bonafide",
                "p_spoof",
                "logit_bonafide",
                "logit_spoof",
            ],
        )
        writer.writeheader()
        writer.writerows(predictions)


def write_predictions_jsonl(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_plots(
    output_dir: Path,
    *,
    split: str,
    metrics: dict[str, Any],
    labels: np.ndarray,
    probs: np.ndarray,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for --plots-dir. Install dependencies from requirements.txt."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    spoof_scores = probs[:, 1]
    unique_labels = np.unique(labels)

    if len(unique_labels) == 2:
        fpr, tpr, _ = roc_curve(labels, spoof_scores, pos_label=1)
        roc_fig, roc_ax = plt.subplots(figsize=(7, 5))
        roc_ax.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.4f}", linewidth=2)
        roc_ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
        roc_ax.set_title(f"{split} ROC Curve")
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.legend(loc="lower right")
        roc_ax.grid(alpha=0.25)
        roc_fig.tight_layout()
        roc_fig.savefig(output_dir / f"{split}_roc_curve.png", dpi=160)
        plt.close(roc_fig)

        pr_precision, pr_recall, _ = precision_recall_curve(labels, spoof_scores, pos_label=1)
        pr_fig, pr_ax = plt.subplots(figsize=(7, 5))
        pr_ax.plot(pr_recall, pr_precision, label=f"AP = {metrics['average_precision']:.4f}", linewidth=2)
        pr_ax.set_title(f"{split} Precision-Recall Curve")
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.legend(loc="lower left")
        pr_ax.grid(alpha=0.25)
        pr_fig.tight_layout()
        pr_fig.savefig(output_dir / f"{split}_precision_recall_curve.png", dpi=160)
        plt.close(pr_fig)

    cm = metrics["confusion_matrix"]["matrix"]
    cm_fig, cm_ax = plt.subplots(figsize=(5.5, 5))
    image = cm_ax.imshow(cm, cmap="Blues")
    cm_ax.set_title(f"{split} Confusion Matrix")
    cm_ax.set_xticks([0, 1], labels=["bonafide", "spoof"])
    cm_ax.set_yticks([0, 1], labels=["bonafide", "spoof"])
    cm_ax.set_xlabel("Predicted")
    cm_ax.set_ylabel("True")
    for row_index, row in enumerate(cm):
        for col_index, value in enumerate(row):
            cm_ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")
    cm_fig.colorbar(image, ax=cm_ax, fraction=0.046, pad=0.04)
    cm_fig.tight_layout()
    cm_fig.savefig(output_dir / f"{split}_confusion_matrix.png", dpi=160)
    plt.close(cm_fig)

    hist_fig, hist_ax = plt.subplots(figsize=(7, 5))
    hist_ax.hist(spoof_scores[labels == 0], bins=50, alpha=0.6, label="bonafide", color="#4c78a8")
    hist_ax.hist(spoof_scores[labels == 1], bins=50, alpha=0.6, label="spoof", color="#e45756")
    hist_ax.axvline(metrics["eer_threshold"], linestyle="--", linewidth=1.5, color="black", label="EER threshold")
    hist_ax.set_title(f"{split} Spoof Score Distribution")
    hist_ax.set_xlabel("p_spoof")
    hist_ax.set_ylabel("Count")
    hist_ax.legend()
    hist_ax.grid(alpha=0.2)
    hist_fig.tight_layout()
    hist_fig.savefig(output_dir / f"{split}_spoof_score_histogram.png", dpi=160)
    plt.close(hist_fig)

    summary_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "eer"]
    summary_values = [float(metrics[name]) for name in summary_names]
    summary_display = summary_values[:-1] + [1.0 - summary_values[-1]]
    summary_labels = ["acc", "prec", "rec", "f1", "auc", "ap", "1-eer"]

    summary_fig, summary_ax = plt.subplots(figsize=(8, 5))
    bars = summary_ax.bar(summary_labels, summary_display, color="#59a14f")
    summary_ax.set_ylim(0.0, 1.0)
    summary_ax.set_title(f"{split} Metrics Summary")
    summary_ax.set_ylabel("Value")
    summary_ax.grid(axis="y", alpha=0.2)
    summary_ax.text(
        0.98,
        0.02,
        f"loss = {metrics['loss']:.4f}",
        transform=summary_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )
    for bar, value in zip(bars, summary_display, strict=True):
        summary_ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    summary_fig.tight_layout()
    summary_fig.savefig(output_dir / f"{split}_metrics_summary.png", dpi=160)
    plt.close(summary_fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, device)
    loader = make_loader(args, checkpoint, model.sample_rate)
    criterion = make_criterion(checkpoint, loader.dataset, device)
    eval_passes = resolve_eval_passes(checkpoint, args.eval_passes)
    use_amp = resolve_use_amp(checkpoint, device, args.mixed_precision)
    collect_prediction_rows = bool(args.predictions_csv or args.predictions_jsonl)
    metrics, predictions, plot_payload = evaluate_with_predictions(
        model,
        loader,
        criterion,
        device,
        use_amp=use_amp,
        eval_passes=eval_passes,
        desc_prefix=args.split,
        collect_prediction_rows=collect_prediction_rows,
    )

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "split": args.split,
        "items": len(loader.dataset),
        "epoch": checkpoint.get("epoch"),
        "best_metric_value": checkpoint.get("best_metric_value"),
        "eval_passes": eval_passes,
        "metrics": metrics,
    }
    if args.plots_dir:
        summary["plots_dir"] = str(Path(args.plots_dir))
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if args.plots_dir:
        render_plots(
            Path(args.plots_dir),
            split=args.split,
            metrics=metrics,
            labels=plot_payload["labels"],
            probs=plot_payload["probs"],
        )
    if args.predictions_csv and predictions is not None:
        write_predictions_csv(Path(args.predictions_csv), predictions)
    if args.predictions_jsonl and predictions is not None:
        write_predictions_jsonl(Path(args.predictions_jsonl), predictions)


if __name__ == "__main__":
    main()
