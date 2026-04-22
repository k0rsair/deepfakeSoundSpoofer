from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepfake_spoofer.data import ASVspoof5Dataset, ID_TO_LABEL, collate_audio, load_audio
from deepfake_spoofer.predict import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit/evaluate a KNN head over neural embeddings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Fit KNN on embeddings from a labeled split.")
    add_shared_dataset_args(fit)
    fit.add_argument("--output", default=None, help="Path to save KNN pickle. Defaults to <checkpoint>.knn.pkl.")
    fit.add_argument("--neighbors", type=int, default=10)
    fit.add_argument("--weights", default="distance", choices=["uniform", "distance"])

    evaluate = subparsers.add_parser("eval", help="Evaluate KNN and neural classifier on a labeled split.")
    add_shared_dataset_args(evaluate)
    evaluate.add_argument("--knn", required=True, help="Path to KNN pickle.")

    predict = subparsers.add_parser("predict", help="Predict audio files with neural and KNN heads.")
    add_shared_model_args(predict)
    predict.add_argument("--knn", required=True, help="Path to KNN pickle.")
    predict.add_argument("--max-seconds", type=float, default=None)
    predict.add_argument("audio", nargs="+")
    return parser.parse_args()


def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    add_shared_model_args(parser)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="train", choices=["train", "dev", "eval"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)


def load_checkpoint_and_model(args: argparse.Namespace) -> tuple[dict[str, Any], torch.nn.Module, torch.device]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    return checkpoint, model, device


def dataset_max_seconds(checkpoint: dict[str, Any], requested: float | None) -> float:
    if requested is not None:
        return requested
    return float(checkpoint.get("config", {}).get("max_seconds", 4.0))


def make_loader(args: argparse.Namespace, checkpoint: dict[str, Any], sample_rate: int) -> DataLoader:
    config = checkpoint.get("config", {})
    ssl_cache_dir = None
    if config.get("freeze_wav2vec") and config.get("model_type", "wav2vec_pyara") in {"fusion", "wav2vec_pyara"}:
        ssl_cache_dir = config.get("wav2vec_cache_dir")
    dataset = ASVspoof5Dataset(
        args.data_dir,
        args.split,
        sample_rate=sample_rate,
        max_seconds=dataset_max_seconds(checkpoint, args.max_seconds),
        random_crop=False,
        limit=args.limit,
        limit_per_class=args.limit_per_class,
        ssl_cache_dir=ssl_cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_audio,
    )


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    logits_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="embeddings", leave=False):
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            targets = batch["labels"].cpu().numpy()
            embedding, logits = model(waveforms, lengths)
            embedding = F.normalize(embedding.float(), p=2, dim=1)
            embeddings.append(embedding.cpu().numpy())
            logits_rows.append(logits.float().cpu().numpy())
            labels.append(targets)

    return np.vstack(embeddings), np.vstack(logits_rows), np.concatenate(labels)


def probability_from_logits(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def metrics_from_probs(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    metrics = {"accuracy": float((pred == labels).mean())}
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
    return metrics


def command_fit(args: argparse.Namespace) -> None:
    checkpoint, model, device = load_checkpoint_and_model(args)
    loader = make_loader(args, checkpoint, model.sample_rate)
    embeddings, _, labels = extract_embeddings(model, loader, device)

    knn = KNeighborsClassifier(n_neighbors=args.neighbors, weights=args.weights)
    knn.fit(embeddings, labels)

    output = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".knn.pkl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(
            {
                "knn": knn,
                "checkpoint": str(Path(args.checkpoint)),
                "split": args.split,
                "embedding_dim": int(embeddings.shape[1]),
                "labels": ID_TO_LABEL,
            },
            handle,
        )
    print(f"saved KNN head to {output}")
    print(f"fit embeddings: {embeddings.shape}, labels: {labels.shape}")


def command_eval(args: argparse.Namespace) -> None:
    checkpoint, model, device = load_checkpoint_and_model(args)
    with Path(args.knn).open("rb") as handle:
        knn_payload = pickle.load(handle)
    knn: KNeighborsClassifier = knn_payload["knn"]

    loader = make_loader(args, checkpoint, model.sample_rate)
    embeddings, logits, labels = extract_embeddings(model, loader, device)
    neural_probs = probability_from_logits(logits)
    knn_probs = knn.predict_proba(embeddings)
    avg_probs = (neural_probs + knn_probs) / 2.0

    print(f"neural: {metrics_from_probs(labels, neural_probs)}")
    print(f"knn:    {metrics_from_probs(labels, knn_probs)}")
    print(f"avg:    {metrics_from_probs(labels, avg_probs)}")


def command_predict(args: argparse.Namespace) -> None:
    checkpoint, model, device = load_checkpoint_and_model(args)
    with Path(args.knn).open("rb") as handle:
        knn_payload = pickle.load(handle)
    knn: KNeighborsClassifier = knn_payload["knn"]

    sample_rate = model.sample_rate
    max_seconds = dataset_max_seconds(checkpoint, args.max_seconds)
    max_samples = int(max_seconds * sample_rate) if max_seconds > 0 else None

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
            embedding, logits = model(waveforms, lengths)
            embedding = F.normalize(embedding.float(), p=2, dim=1).cpu().numpy()
            neural_probs = torch.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()
            knn_probs = knn.predict_proba(embedding).squeeze(0)
            avg_probs = (neural_probs + knn_probs) / 2.0
            pred_id = int(avg_probs.argmax())
            print(
                f"{audio_path}\tlabel={ID_TO_LABEL[pred_id]}\t"
                f"neural_spoof={neural_probs[1]:.6f}\t"
                f"knn_spoof={knn_probs[1]:.6f}\t"
                f"avg_spoof={avg_probs[1]:.6f}"
            )


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        command_fit(args)
    elif args.command == "eval":
        command_eval(args)
    elif args.command == "predict":
        command_predict(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
