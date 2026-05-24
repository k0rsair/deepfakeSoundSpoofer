from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from deepfake_spoofer.data import ID_TO_LABEL, load_audio
from deepfake_spoofer.model import build_spoof_detector


@dataclass(frozen=True)
class PredictionResult:
    path: str
    label_id: int
    label: str
    p_bonafide: float
    p_spoof: float
    confidence: float
    sample_rate: int
    original_seconds: float
    processed_seconds: float
    cropped: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether audio is bonafide or spoof.")
    parser.add_argument("audio", nargs="+", help="Path(s) to audio files.")
    parser.add_argument("--checkpoint", default="runs/wav2vec_pyara/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seconds", type=float, default=None)
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    config = checkpoint.get("config", {})
    model = build_spoof_detector(
        model_type=config.get("model_type", "wav2vec_pyara"),
        bundle_name=config.get("bundle", "WAV2VEC2_XLSR_300M"),
        freeze_wav2vec=True,
        freeze_feature_extractor=not config.get("unfreeze_feature_extractor", False),
        freeze_transformer_layers=int(config.get("freeze_transformer_layers", 0)),
        wav2vec_layers=config.get("wav2vec_layers"),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def load_checkpoint_and_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, device)
    return checkpoint, model


def checkpoint_max_seconds(checkpoint: dict[str, Any], requested: float | None) -> float:
    if requested is not None:
        return requested
    return float(checkpoint.get("config", {}).get("max_seconds", 4.0))


def predict_audio_path(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    audio_path: str | Path,
    *,
    max_seconds: float | None = None,
) -> PredictionResult:
    sample_rate = model.sample_rate
    resolved_max_seconds = checkpoint_max_seconds(checkpoint, max_seconds)
    max_samples = int(resolved_max_seconds * sample_rate) if resolved_max_seconds > 0 else None

    audio = load_audio(
        Path(audio_path),
        sample_rate=sample_rate,
        max_samples=max_samples,
        random_crop=False,
    )
    waveforms = audio.waveform.unsqueeze(0).to(device)
    lengths = torch.tensor([audio.valid_length], dtype=torch.long, device=device)
    with torch.no_grad():
        _, logits = model(waveforms, lengths)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_id = int(probs.argmax().item())
    p_bonafide = float(probs[0].item())
    p_spoof = float(probs[1].item())
    original_seconds = audio.original_length / sample_rate
    processed_seconds = audio.valid_length / sample_rate
    return PredictionResult(
        path=str(audio_path),
        label_id=pred_id,
        label=ID_TO_LABEL[pred_id],
        p_bonafide=p_bonafide,
        p_spoof=p_spoof,
        confidence=max(p_bonafide, p_spoof),
        sample_rate=sample_rate,
        original_seconds=original_seconds,
        processed_seconds=processed_seconds,
        cropped=audio.original_length != audio.valid_length,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint, model = load_checkpoint_and_model(args.checkpoint, device)

    for audio_path in args.audio:
        result = predict_audio_path(
            checkpoint,
            model,
            device,
            audio_path,
            max_seconds=args.max_seconds,
        )
        print(
            f"{audio_path}\tlabel={result.label}\t"
            f"p_bonafide={result.p_bonafide:.6f}\tp_spoof={result.p_spoof:.6f}"
        )


if __name__ == "__main__":
    main()
