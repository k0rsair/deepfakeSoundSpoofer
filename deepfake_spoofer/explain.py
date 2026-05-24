from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from deepfake_spoofer.data import ID_TO_LABEL, load_audio
from deepfake_spoofer.predict import checkpoint_max_seconds, load_checkpoint_and_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a single prediction with time occlusion and branch ablation.")
    parser.add_argument("audio", help="Path to an audio file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--window-seconds", type=float, default=0.4)
    parser.add_argument("--hop-seconds", type=float, default=0.2)
    parser.add_argument("--baseline", default="zero", choices=["zero", "mean"], help="Occlusion replacement value.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--lime-samples", type=int, default=0, help="Run LIME-style random segment masking.")
    parser.add_argument("--lime-segments", type=int, default=20, help="Number of contiguous audio segments for LIME.")
    parser.add_argument("--lime-batch-size", type=int, default=16)
    parser.add_argument("--lime-alpha", type=float, default=1.0, help="Ridge regularization for the local surrogate.")
    parser.add_argument("--lime-seed", type=int, default=42)
    parser.add_argument("--lime-output-csv", default=None)
    parser.add_argument("--plots-dir", default=None)
    return parser.parse_args()


def probability_from_logits(logits: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits.float(), dim=1).squeeze(0)
    logits = logits.float().squeeze(0)
    return {
        "p_bonafide": float(probs[0].item()),
        "p_spoof": float(probs[1].item()),
        "logit_bonafide": float(logits[0].item()),
        "logit_spoof": float(logits[1].item()),
        "logit_margin_spoof": float((logits[1] - logits[0]).item()),
        "pred_label": int(probs.argmax().item()),
    }


@torch.no_grad()
def score_waveform(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    length: int,
    device: torch.device,
) -> dict[str, float]:
    waveforms = waveform.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    _, logits = model(waveforms, lengths)
    return probability_from_logits(logits)


def occlude_waveform(
    waveform: torch.Tensor,
    *,
    start_sample: int,
    end_sample: int,
    baseline: str,
) -> torch.Tensor:
    occluded = waveform.clone()
    if baseline == "mean":
        replacement = float(waveform.mean().item())
    else:
        replacement = 0.0
    occluded[start_sample:end_sample] = replacement
    return occluded


def replacement_value(waveform: torch.Tensor, baseline: str) -> float:
    if baseline == "mean":
        return float(waveform.mean().item())
    return 0.0


@torch.no_grad()
def branch_ablation_scores(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    length: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if not all(hasattr(model, attr) for attr in ["wav2vec_branch", "mfcc_branch", "fusion_classifier"]):
        return None

    waveforms = waveform.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    wav_embedding, _ = model.wav2vec_branch(waveforms, lengths)
    mfcc_embedding, _ = model.mfcc_branch(waveforms, lengths)

    full_embedding = torch.cat([wav_embedding, mfcc_embedding], dim=1)
    zero_wav_embedding = torch.cat([torch.zeros_like(wav_embedding), mfcc_embedding], dim=1)
    zero_mfcc_embedding = torch.cat([wav_embedding, torch.zeros_like(mfcc_embedding)], dim=1)

    full = probability_from_logits(model.fusion_classifier(full_embedding))
    without_wav2vec = probability_from_logits(model.fusion_classifier(zero_wav_embedding))
    without_mfcc = probability_from_logits(model.fusion_classifier(zero_mfcc_embedding))

    return {
        "full_from_branches": full,
        "without_wav2vec": without_wav2vec,
        "without_mfcc": without_mfcc,
        "delta_without_wav2vec": full["p_spoof"] - without_wav2vec["p_spoof"],
        "delta_without_mfcc": full["p_spoof"] - without_mfcc["p_spoof"],
        "delta_margin_without_wav2vec": full["logit_margin_spoof"] - without_wav2vec["logit_margin_spoof"],
        "delta_margin_without_mfcc": full["logit_margin_spoof"] - without_mfcc["logit_margin_spoof"],
    }


@torch.no_grad()
def score_waveform_batch(
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    length: int,
    device: torch.device,
) -> list[dict[str, float]]:
    lengths = torch.full((waveforms.size(0),), int(length), dtype=torch.long, device=device)
    _, logits = model(waveforms.to(device), lengths)
    probs = torch.softmax(logits.float(), dim=1)
    margins = logits.float()[:, 1] - logits.float()[:, 0]
    rows: list[dict[str, float]] = []
    for index in range(waveforms.size(0)):
        rows.append(
            {
                "p_bonafide": float(probs[index, 0].item()),
                "p_spoof": float(probs[index, 1].item()),
                "logit_margin_spoof": float(margins[index].item()),
            }
        )
    return rows


def lime_segment_bounds(total_samples: int, segment_count: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, total_samples, segment_count + 1, dtype=np.int64)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(segment_count)]


def apply_segment_mask(
    waveform: torch.Tensor,
    bounds: list[tuple[int, int]],
    mask: np.ndarray,
    *,
    baseline: str,
) -> torch.Tensor:
    masked = waveform.clone()
    replacement = replacement_value(waveform, baseline)
    for keep, (start, end) in zip(mask, bounds, strict=True):
        if not keep:
            masked[start:end] = replacement
    return masked


def lime_segment_explanation(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    length: int,
    device: torch.device,
    *,
    sample_rate: int,
    segment_count: int,
    sample_count: int,
    batch_size: int,
    alpha: float,
    seed: int,
    baseline: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    segment_count = max(2, min(segment_count, int(waveform.numel())))
    sample_count = max(sample_count, segment_count + 2)
    bounds = lime_segment_bounds(int(waveform.numel()), segment_count)

    masks = rng.integers(0, 2, size=(sample_count, segment_count), endpoint=False).astype(bool)
    masks[0, :] = True
    for index in range(segment_count):
        masks[index + 1, :] = True
        masks[index + 1, index] = False
    empty_masks = masks.sum(axis=1) == 0
    masks[empty_masks, rng.integers(0, segment_count, size=int(empty_masks.sum()))] = True

    scores: list[dict[str, float]] = []
    for start in range(0, sample_count, batch_size):
        batch_masks = masks[start : start + batch_size]
        batch_waveforms = torch.stack(
            [
                apply_segment_mask(waveform, bounds, mask, baseline=baseline)
                for mask in batch_masks
            ],
            dim=0,
        )
        scores.extend(score_waveform_batch(model, batch_waveforms, length, device))

    y_margin = np.asarray([row["logit_margin_spoof"] for row in scores], dtype=np.float64)
    x = masks.astype(np.float64)
    distances = 1.0 - (x.sum(axis=1) / max(segment_count, 1))
    kernel_width = 0.25
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    regressor = Ridge(alpha=alpha)
    regressor.fit(x, y_margin, sample_weight=weights)
    y_pred = regressor.predict(x)
    r2 = float(r2_score(y_margin, y_pred, sample_weight=weights))

    rows: list[dict[str, Any]] = []
    for index, ((start_sample, end_sample), coefficient) in enumerate(zip(bounds, regressor.coef_, strict=True)):
        rows.append(
            {
                "segment_index": index,
                "start_seconds": start_sample / sample_rate,
                "end_seconds": end_sample / sample_rate,
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
                "effect": "supports_spoof" if coefficient > 0 else "supports_bonafide",
            }
        )

    rows_sorted = sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)
    return {
        "samples": int(sample_count),
        "segments": int(segment_count),
        "alpha": float(alpha),
        "seed": int(seed),
        "weighted_r2": r2,
        "intercept": float(regressor.intercept_),
        "top_segments": rows_sorted[:10],
        "segments_table": rows,
    }


def build_occlusion_rows(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    length: int,
    device: torch.device,
    *,
    sample_rate: int,
    base_p_spoof: float,
    base_logit_margin: float,
    window_seconds: float,
    hop_seconds: float,
    baseline: str,
) -> list[dict[str, Any]]:
    window_samples = max(int(window_seconds * sample_rate), 1)
    hop_samples = max(int(hop_seconds * sample_rate), 1)
    total_samples = int(waveform.numel())
    rows: list[dict[str, Any]] = []

    starts = list(range(0, max(total_samples - window_samples + 1, 1), hop_samples))
    if not starts or starts[-1] + window_samples < total_samples:
        starts.append(max(total_samples - window_samples, 0))

    for start in starts:
        end = min(start + window_samples, total_samples)
        occluded = occlude_waveform(waveform, start_sample=start, end_sample=end, baseline=baseline)
        score = score_waveform(model, occluded, length, device)
        p_spoof = score["p_spoof"]
        margin = score["logit_margin_spoof"]
        rows.append(
            {
                "start_seconds": start / sample_rate,
                "end_seconds": end / sample_rate,
                "p_spoof_after_occlusion": p_spoof,
                "delta_p_spoof": base_p_spoof - p_spoof,
                "abs_delta_p_spoof": abs(base_p_spoof - p_spoof),
                "logit_margin_after_occlusion": margin,
                "delta_logit_margin_spoof": base_logit_margin - margin,
                "abs_delta_logit_margin_spoof": abs(base_logit_margin - margin),
                "pred_label_after_occlusion": score["pred_label"],
                "pred_text_after_occlusion": ID_TO_LABEL[score["pred_label"]],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_plot(path: Path, rows: list[dict[str, Any]], *, base_p_spoof: float, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    starts = np.asarray([row["start_seconds"] for row in rows], dtype=np.float64)
    ends = np.asarray([row["end_seconds"] for row in rows], dtype=np.float64)
    centers = (starts + ends) / 2.0
    deltas = np.asarray([row["delta_p_spoof"] for row in rows], dtype=np.float64)
    margin_deltas = np.asarray([row["delta_logit_margin_spoof"] for row in rows], dtype=np.float64)
    after_scores = np.asarray([row["p_spoof_after_occlusion"] for row in rows], dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(centers, margin_deltas, width=np.maximum(ends - starts, 0.01), color="#f28e2b", alpha=0.85)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("delta logit margin")
    axes[0].set_title("Time occlusion importance")
    axes[0].grid(alpha=0.2)

    axes[1].bar(centers, deltas, width=np.maximum(ends - starts, 0.01), color="#59a14f", alpha=0.85)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("delta p_spoof")
    axes[1].grid(alpha=0.2)

    axes[2].plot(centers, after_scores, color="#4c78a8", linewidth=2)
    axes[2].axhline(base_p_spoof, color="black", linestyle="--", linewidth=1.2, label=f"base={base_p_spoof:.3f}")
    axes[2].axhline(threshold, color="#e45756", linestyle=":", linewidth=1.2, label=f"threshold={threshold:.3f}")
    axes[2].set_xlabel("seconds")
    axes[2].set_ylabel("p_spoof after occlusion")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.2)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_lime_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)[:20]
    labels = [f"{row['start_seconds']:.2f}-{row['end_seconds']:.2f}s" for row in sorted_rows]
    values = [row["coefficient"] for row in sorted_rows]
    colors = ["#e45756" if value > 0 else "#4c78a8" for value in values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(sorted_rows) * 0.35)))
    positions = np.arange(len(sorted_rows))
    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("LIME-style segment coefficients")
    ax.set_xlabel("local surrogate coefficient for spoof margin")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def natural_language_summary(payload: dict[str, Any]) -> str:
    base = payload["base_prediction"]
    label = ID_TO_LABEL[int(base["pred_label"])]
    top_support = payload["top_supporting_spoof_windows"][:3]
    top_against = payload["top_opposing_spoof_windows"][:3]
    branch = payload.get("branch_ablation")
    lime = payload.get("lime")

    parts = [
        f"Prediction: {label}, p_spoof={base['p_spoof']:.4f}, threshold={payload['threshold']:.4f}.",
    ]
    if top_support:
        windows = ", ".join(
            f"{row['start_seconds']:.2f}-{row['end_seconds']:.2f}s "
            f"(margin_delta={row['delta_logit_margin_spoof']:.3f})"
            for row in top_support
        )
        parts.append(f"Most spoof-supporting windows: {windows}.")
    if top_against:
        windows = ", ".join(
            f"{row['start_seconds']:.2f}-{row['end_seconds']:.2f}s "
            f"(margin_delta={row['delta_logit_margin_spoof']:.3f})"
            for row in top_against
        )
        if all(row["delta_logit_margin_spoof"] >= 0 for row in top_against):
            parts.append(f"Least spoof-supporting windows: {windows}.")
        else:
            parts.append(f"Windows that most opposed spoof: {windows}.")
    if branch is not None:
        parts.append(
            "Branch ablation: "
            f"without wav2vec p_spoof={branch['without_wav2vec']['p_spoof']:.4f}, "
            f"without MFCC p_spoof={branch['without_mfcc']['p_spoof']:.4f}; "
            f"margin drops: wav2vec={branch['delta_margin_without_wav2vec']:.3f}, "
            f"MFCC={branch['delta_margin_without_mfcc']:.3f}."
        )
    if lime is not None:
        top_lime = lime["top_segments"][:3]
        segments = ", ".join(
            f"{row['start_seconds']:.2f}-{row['end_seconds']:.2f}s "
            f"(coef={row['coefficient']:.3f})"
            for row in top_lime
        )
        parts.append(f"LIME-style surrogate weighted R2={lime['weighted_r2']:.3f}; top segments: {segments}.")
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint, model = load_checkpoint_and_model(args.checkpoint, device)
    sample_rate = model.sample_rate
    max_seconds = checkpoint_max_seconds(checkpoint, args.max_seconds)
    max_samples = int(max_seconds * sample_rate) if max_seconds > 0 else None

    audio = load_audio(
        Path(args.audio),
        sample_rate=sample_rate,
        max_samples=max_samples,
        random_crop=False,
    )
    waveform = audio.waveform
    base_prediction = score_waveform(model, waveform, audio.valid_length, device)
    occlusion_rows = build_occlusion_rows(
        model,
        waveform,
        audio.valid_length,
        device,
        sample_rate=sample_rate,
        base_p_spoof=base_prediction["p_spoof"],
        base_logit_margin=base_prediction["logit_margin_spoof"],
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        baseline=args.baseline,
    )

    top_supporting = sorted(occlusion_rows, key=lambda row: row["delta_logit_margin_spoof"], reverse=True)[:10]
    top_opposing = sorted(occlusion_rows, key=lambda row: row["delta_logit_margin_spoof"])[:10]
    branch_ablation = branch_ablation_scores(model, waveform, audio.valid_length, device)
    lime = None
    if args.lime_samples > 0:
        lime = lime_segment_explanation(
            model,
            waveform,
            audio.valid_length,
            device,
            sample_rate=sample_rate,
            segment_count=args.lime_segments,
            sample_count=args.lime_samples,
            batch_size=args.lime_batch_size,
            alpha=args.lime_alpha,
            seed=args.lime_seed,
            baseline=args.baseline,
        )

    payload: dict[str, Any] = {
        "audio": str(Path(args.audio)),
        "checkpoint": str(Path(args.checkpoint)),
        "model_type": checkpoint.get("config", {}).get("model_type"),
        "sample_rate": sample_rate,
        "original_seconds": audio.original_length / sample_rate,
        "processed_seconds": audio.valid_length / sample_rate,
        "threshold": args.threshold,
        "base_prediction": {
            **base_prediction,
            "pred_text": ID_TO_LABEL[base_prediction["pred_label"]],
        },
        "occlusion": {
            "window_seconds": args.window_seconds,
            "hop_seconds": args.hop_seconds,
            "baseline": args.baseline,
        },
        "top_supporting_spoof_windows": top_supporting,
        "top_opposing_spoof_windows": top_opposing,
        "branch_ablation": branch_ablation,
        "lime": None if lime is None else {key: value for key, value in lime.items() if key != "segments_table"},
    }
    payload["explanation"] = natural_language_summary(payload)

    if args.output_csv:
        write_csv(Path(args.output_csv), occlusion_rows)
    if args.lime_output_csv and lime is not None:
        write_csv(Path(args.lime_output_csv), lime["segments_table"])
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.plots_dir:
        render_plot(
            Path(args.plots_dir) / "time_occlusion.png",
            occlusion_rows,
            base_p_spoof=base_prediction["p_spoof"],
            threshold=args.threshold,
        )
        if lime is not None:
            render_lime_plot(Path(args.plots_dir) / "lime_segments.png", lime["segments_table"])

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
