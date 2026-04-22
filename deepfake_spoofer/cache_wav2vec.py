from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepfake_spoofer.data import ASVspoof5Dataset, collate_audio, wav2vec_cache_path
from deepfake_spoofer.model import Wav2VecFrontend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen wav2vec features for later training.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bundle", default="WAV2VEC2_XLSR_300M")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev"],
        choices=["train", "dev", "eval"],
        help="Dataset splits to cache. Eval is large; include it only if you really need it.",
    )
    parser.add_argument(
        "--wav2vec-layers",
        type=int,
        default=21,
        help="Cache exactly this wav2vec layer via extract_features(..., num_layers=N).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument(
        "--sort-by-length",
        default="descending",
        choices=["none", "descending", "ascending"],
        help="Sort files by duration before caching. Descending stabilizes GPU peak earlier.",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=25,
        help="Call torch.cuda.empty_cache() every N batches. Set 0 to disable.",
    )
    parser.add_argument(
        "--log-memory-every",
        type=int,
        default=25,
        help="Print allocated/reserved/max_reserved CUDA memory every N batches. Set 0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tensor_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.float32


def save_meta(output_dir: Path, frontend: Wav2VecFrontend, args: argparse.Namespace) -> None:
    meta = {
        "bundle": args.bundle,
        "wav2vec_layers": args.wav2vec_layers,
        "sample_rate": frontend.sample_rate,
        "feature_dim": frontend.out_dim,
        "dtype": args.dtype,
        "splits": args.splits,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def reorder_dataset_by_duration(dataset: ASVspoof5Dataset, split: str, order: str) -> None:
    if order == "none":
        return

    descending = order == "descending"
    durations: list[tuple[int, int]] = []
    for index, item in enumerate(tqdm(dataset.items, desc=f"probe:{split}", leave=False)):
        info = sf.info(str(item.path))
        durations.append((index, int(info.frames)))
    durations.sort(key=lambda row: row[1], reverse=descending)
    dataset.items = [dataset.items[index] for index, _ in durations]


def maybe_log_cuda_memory(args: argparse.Namespace, split: str, batch_index: int) -> None:
    if args.device != "cuda" or args.log_memory_every <= 0 or batch_index % args.log_memory_every != 0:
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(
        f"{split}: batch={batch_index} "
        f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB max_reserved={max_reserved:.2f}GiB"
    )


def maybe_release_cuda_cache(args: argparse.Namespace, batch_index: int) -> None:
    if args.device == "cuda" and args.empty_cache_every > 0 and batch_index % args.empty_cache_every == 0:
        torch.cuda.empty_cache()
        gc.collect()


def cache_split(args: argparse.Namespace, frontend: Wav2VecFrontend, split: str, output_dir: Path) -> None:
    dataset = ASVspoof5Dataset(
        args.data_dir,
        split,
        sample_rate=frontend.sample_rate,
        max_seconds=0,
        random_crop=False,
        limit=args.limit,
        limit_per_class=args.limit_per_class,
    )
    reorder_dataset_by_duration(dataset, split, args.sort_by_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_audio,
    )

    target_dtype = tensor_dtype(args.dtype)
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    cached = 0
    skipped = 0
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"cache:{split}", leave=False), start=1):
            file_ids = batch["file_ids"]
            output_paths = [wav2vec_cache_path(output_dir, split, file_id) for file_id in file_ids]
            if not args.overwrite and all(path.exists() for path in output_paths):
                skipped += len(output_paths)
                maybe_log_cuda_memory(args, split, batch_index)
                maybe_release_cuda_cache(args, batch_index)
                continue

            waveforms = batch["waveforms"].to(args.device, non_blocking=True)
            lengths = batch["lengths"].to(args.device, non_blocking=True)
            features, feature_lengths = frontend(waveforms, lengths)
            features = features.detach().cpu()
            feature_lengths = feature_lengths.detach().cpu() if feature_lengths is not None else None

            for index, file_id in enumerate(file_ids):
                output_path = output_paths[index]
                if output_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                current_feature_length = int(feature_lengths[index]) if feature_lengths is not None else int(features[index].size(0))
                payload = {
                    "file_id": file_id,
                    "bundle": args.bundle,
                    "wav2vec_layers": args.wav2vec_layers,
                    "raw_num_samples": int(batch["lengths"][index]),
                    "features": features[index, :current_feature_length].to(dtype=target_dtype).contiguous(),
                }
                torch.save(payload, output_path)
                cached += 1

            del waveforms, lengths, features, feature_lengths, output_paths, file_ids
            maybe_log_cuda_memory(args, split, batch_index)
            maybe_release_cuda_cache(args, batch_index)

    print(f"{split}: cached={cached} skipped={skipped} total={len(dataset)}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frontend = Wav2VecFrontend(
        bundle_name=args.bundle,
        freeze_wav2vec=True,
        wav2vec_layers=args.wav2vec_layers,
    ).to(args.device)
    frontend.eval()
    save_meta(output_dir, frontend, args)

    for split in args.splits:
        cache_split(args, frontend, split, output_dir)


if __name__ == "__main__":
    main()
