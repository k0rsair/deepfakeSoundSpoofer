from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROTOCOL_COLUMNS = [
    "speaker_id",
    "file_id",
    "gender",
    "codec",
    "codec_level",
    "source_file",
    "condition",
    "attack",
    "label_text",
    "extra",
]

LABEL_TO_ID = {"bonafide": 0, "spoof": 1}
ID_TO_LABEL = {0: "bonafide", 1: "spoof"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interpretation reports from saved prediction scores.")
    parser.add_argument("--predictions-csv", required=True, help="CSV produced by deepfake_spoofer.evaluate.")
    parser.add_argument("--protocol", required=True, help="ASVspoof5 protocol TSV matching the evaluated split.")
    parser.add_argument("--output-dir", required=True, help="Directory for interpretation artifacts.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Spoof decision threshold.")
    parser.add_argument("--min-group-count", type=int, default=100, help="Hide smaller groups from grouped reports.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of example rows per examples file.")
    parser.add_argument(
        "--target-fpr",
        type=float,
        nargs="*",
        default=[0.05, 0.10, 0.15, 0.20],
        help="Bonafide false-positive targets for threshold suggestions.",
    )
    parser.add_argument("--plots-dir", default=None, help="Optional directory for interpretation PNG plots.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot rendering for faster reports.")
    return parser.parse_args()


def read_protocol(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            padded = parts[: len(PROTOCOL_COLUMNS)] + [""] * max(len(PROTOCOL_COLUMNS) - len(parts), 0)
            row = dict(zip(PROTOCOL_COLUMNS, padded, strict=True))
            file_id = row["file_id"]
            if row["label_text"].lower() in LABEL_TO_ID:
                row["label"] = str(LABEL_TO_ID[row["label_text"].lower()])
            rows[file_id] = row
    return rows


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "label": int(row["label"]),
                    "pred_label": int(row["pred_label"]),
                    "p_bonafide": float(row["p_bonafide"]),
                    "p_spoof": float(row["p_spoof"]),
                    "logit_bonafide": float(row["logit_bonafide"]),
                    "logit_spoof": float(row["logit_spoof"]),
                }
            )
    return rows


def merge_protocol(predictions: Iterable[dict[str, Any]], protocol_rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in predictions:
        protocol = protocol_rows.get(str(row["file_id"]), {})
        merged.append({**protocol, **row})
    return merged


def confusion_counts(labels: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(((labels == 0) & (pred == 0)).sum()),
        "fp": int(((labels == 0) & (pred == 1)).sum()),
        "fn": int(((labels == 1) & (pred == 0)).sum()),
        "tp": int(((labels == 1) & (pred == 1)).sum()),
    }


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    include_ranking: bool = True,
) -> dict[str, Any]:
    pred = (scores >= threshold).astype(np.int64)
    counts = confusion_counts(labels, pred)
    bonafide_total = counts["tn"] + counts["fp"]
    spoof_total = counts["tp"] + counts["fn"]
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "items": int(labels.size),
        **counts,
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "bonafide_fpr": safe_rate(counts["fp"], bonafide_total),
        "spoof_fnr": safe_rate(counts["fn"], spoof_total),
    }
    if include_ranking and len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    elif include_ranking:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def threshold_sweep(labels: np.ndarray, scores: np.ndarray, *, steps: int = 1001) -> list[dict[str, Any]]:
    thresholds = np.linspace(0.0, 1.0, steps)
    return [metrics_at_threshold(labels, scores, float(threshold), include_ranking=False) for threshold in thresholds]


def find_eer_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) != 2:
        return {"eer": float("nan"), "threshold": float("nan")}
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return {"eer": float((fpr[index] + fnr[index]) / 2.0), "threshold": float(thresholds[index])}


def threshold_for_target_fpr(sweep: list[dict[str, Any]], target_fpr: float) -> dict[str, Any]:
    candidates = [row for row in sweep if row["bonafide_fpr"] <= target_fpr]
    if not candidates:
        return {"target_fpr": target_fpr, "threshold": float("nan")}
    best = max(candidates, key=lambda row: (row["recall"], row["f1"], -row["threshold"]))
    return {"target_fpr": float(target_fpr), **best}


def exact_threshold_for_target_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, Any]:
    bonafide_scores = np.sort(scores[labels == 0])
    if bonafide_scores.size == 0:
        return {"target_fpr": float(target_fpr), "threshold": float("nan")}

    allowed_fp = int(np.floor(float(target_fpr) * bonafide_scores.size))
    if allowed_fp <= 0:
        threshold = float(np.nextafter(bonafide_scores[-1], np.inf))
    elif allowed_fp >= bonafide_scores.size:
        threshold = float(0.0)
    else:
        threshold = float(np.nextafter(bonafide_scores[-allowed_fp], np.inf))
    metrics = metrics_at_threshold(labels, scores, threshold, include_ranking=False)
    return {"target_fpr": float(target_fpr), **metrics}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_report(rows: list[dict[str, Any]], group_key: str, *, threshold: float, min_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "") or "-")].append(row)

    report: list[dict[str, Any]] = []
    for value, group_rows in grouped.items():
        if len(group_rows) < min_count:
            continue
        labels = np.asarray([row["label"] for row in group_rows], dtype=np.int64)
        scores = np.asarray([row["p_spoof"] for row in group_rows], dtype=np.float64)
        metrics = metrics_at_threshold(labels, scores, threshold, include_ranking=False)
        report.append(
            {
                "group": group_key,
                "value": value,
                "items": metrics["items"],
                "bonafide": int((labels == 0).sum()),
                "spoof": int((labels == 1).sum()),
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "bonafide_fpr": metrics["bonafide_fpr"],
                "spoof_fnr": metrics["spoof_fnr"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mean_p_spoof": float(scores.mean()),
                "p95_p_spoof": float(np.quantile(scores, 0.95)),
            }
        )
    return sorted(report, key=lambda row: (np.nan_to_num(row["bonafide_fpr"], nan=-1.0), row["items"]), reverse=True)


def example_rows(rows: list[dict[str, Any]], *, threshold: float, top_k: int) -> dict[str, list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        pred = 1 if row["p_spoof"] >= threshold else 0
        margin = abs(row["p_spoof"] - threshold)
        enriched.append(
            {
                **row,
                "threshold_pred_label": pred,
                "threshold_pred_text": ID_TO_LABEL[pred],
                "score_margin": margin,
            }
        )

    false_positives = [row for row in enriched if row["label"] == 0 and row["threshold_pred_label"] == 1]
    false_negatives = [row for row in enriched if row["label"] == 1 and row["threshold_pred_label"] == 0]
    borderline = sorted(enriched, key=lambda row: row["score_margin"])
    confident_errors = sorted(
        false_positives + false_negatives,
        key=lambda row: abs(row["p_spoof"] - (1.0 - row["label"])),
        reverse=True,
    )
    return {
        "false_positives": sorted(false_positives, key=lambda row: row["p_spoof"], reverse=True)[:top_k],
        "false_negatives": sorted(false_negatives, key=lambda row: row["p_spoof"])[:top_k],
        "borderline": borderline[:top_k],
        "confident_errors": confident_errors[:top_k],
    }


def render_plots(
    plots_dir: Path,
    *,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sweep: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 0], bins=60, alpha=0.65, label="bonafide", color="#4c78a8")
    ax.hist(scores[labels == 1], bins=60, alpha=0.55, label="spoof", color="#e45756")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.3f}")
    ax.set_title("Spoof score distribution")
    ax.set_xlabel("p_spoof")
    ax.set_ylabel("count")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "score_distribution.png", dpi=160)
    plt.close(fig)

    thresholds = [row["threshold"] for row in sweep]
    fprs = [row["bonafide_fpr"] for row in sweep]
    fnrs = [row["spoof_fnr"] for row in sweep]
    f1s = [row["f1"] for row in sweep]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, fprs, label="bonafide FPR", linewidth=2)
    ax.plot(thresholds, fnrs, label="spoof FNR", linewidth=2)
    ax.plot(thresholds, f1s, label="F1", linewidth=2)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Threshold tradeoff")
    ax.set_xlabel("p_spoof threshold")
    ax.set_ylabel("metric value")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "threshold_tradeoff.png", dpi=160)
    plt.close(fig)

    for group_key, report in grouped.items():
        filtered = [row for row in report if row["bonafide"] > 0][:20]
        if not filtered:
            continue
        labels_for_plot = [str(row["value"]) for row in filtered]
        values = [row["bonafide_fpr"] for row in filtered]
        fig, ax = plt.subplots(figsize=(9, max(4, len(filtered) * 0.35)))
        positions = np.arange(len(filtered))
        ax.barh(positions, values, color="#f28e2b")
        ax.set_yticks(positions, labels=labels_for_plot)
        ax.invert_yaxis()
        ax.set_xlim(0.0, min(max(values + [0.01]) * 1.15, 1.0))
        ax.set_title(f"Bonafide FPR by {group_key}")
        ax.set_xlabel("false positive rate")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{group_key}_bonafide_fpr.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions_csv)
    protocol_path = Path(args.protocol)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"

    protocol_rows = read_protocol(protocol_path)
    rows = merge_protocol(read_predictions(predictions_path), protocol_rows)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    scores = np.asarray([row["p_spoof"] for row in rows], dtype=np.float64)

    current_metrics = metrics_at_threshold(labels, scores, args.threshold)
    sweep = threshold_sweep(labels, scores)
    best_f1 = max(sweep, key=lambda row: row["f1"])
    eer = find_eer_threshold(labels, scores)
    target_thresholds = [exact_threshold_for_target_fpr(labels, scores, target) for target in args.target_fpr]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for group_key in ["gender", "condition", "attack", "codec", "codec_level"]:
        grouped[group_key] = group_report(rows, group_key, threshold=args.threshold, min_count=args.min_group_count)

    examples = example_rows(rows, threshold=args.threshold, top_k=args.top_k)

    summary = {
        "predictions_csv": str(predictions_path),
        "protocol": str(protocol_path),
        "items": len(rows),
        "threshold": args.threshold,
        "metrics": current_metrics,
        "best_f1_threshold": best_f1,
        "eer": eer,
        "target_fpr_thresholds": target_thresholds,
        "outputs": {
            "groups_csv": "groups.csv",
            "threshold_sweep_csv": "threshold_sweep.csv",
            "false_positives_csv": "false_positives.csv",
            "false_negatives_csv": "false_negatives.csv",
            "borderline_csv": "borderline.csv",
            "confident_errors_csv": "confident_errors.csv",
        },
    }
    if not args.no_plots:
        summary["outputs"]["plots_dir"] = str(plots_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "interpret_summary.json", summary)
    write_csv(output_dir / "threshold_sweep.csv", sweep)
    all_group_rows = [row for report in grouped.values() for row in report]
    write_csv(output_dir / "groups.csv", all_group_rows)
    for name, example_set in examples.items():
        write_csv(output_dir / f"{name}.csv", example_set)
    if not args.no_plots:
        render_plots(
            plots_dir,
            rows=rows,
            labels=labels,
            scores=scores,
            threshold=args.threshold,
            sweep=sweep,
            grouped=grouped,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
