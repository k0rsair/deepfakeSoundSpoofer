from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deepfake_spoofer.data import ID_TO_LABEL, load_audio
from deepfake_spoofer.model import build_spoof_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether audio is bonafide or spoof.")
    parser.add_argument("audio", nargs="+", help="Path(s) to audio files.")
    parser.add_argument("--checkpoint", default="runs/wav2vec_pyara/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seconds", type=float, default=4.0)
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, device)
    sample_rate = model.sample_rate
    max_samples = int(args.max_seconds * sample_rate) if args.max_seconds > 0 else None

    with torch.no_grad():
        for audio_path in args.audio:
            audio = load_audio(
                Path(audio_path),
                sample_rate=sample_rate,
                max_samples=max_samples,
                random_crop=False,
            )
            waveforms = audio.waveform.unsqueeze(0).to(device)
            lengths = torch.tensor([audio.valid_length], dtype=torch.long, device=device)
            _, logits = model(waveforms, lengths)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_id = int(probs.argmax().item())
            print(
                f"{audio_path}\tlabel={ID_TO_LABEL[pred_id]}\t"
                f"p_bonafide={probs[0].item():.6f}\tp_spoof={probs[1].item():.6f}"
            )


if __name__ == "__main__":
    main()
