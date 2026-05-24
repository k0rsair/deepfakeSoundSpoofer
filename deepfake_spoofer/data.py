from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torchaudio.functional as AF
import soundfile as sf
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence


LABEL_TO_ID = {"bonafide": 0, "spoof": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


DEFAULT_PROTOCOLS = {
    "train": "ASVspoof5.train.tsv",
    "dev": "ASVspoof5.dev.track_1.tsv",
    "eval": "ASVspoof5.eval.track_1.tsv",
}


@dataclass(frozen=True)
class AudioItem:
    path: Path
    label: int
    file_id: str
    speaker_id: str


@dataclass(frozen=True)
class AudioLoadResult:
    waveform: torch.Tensor
    valid_length: int
    crop_start: int
    original_length: int


def compute_crop_window(
    raw_num_samples: int,
    *,
    max_samples: int | None,
    random_crop: bool,
    crop_start: int | None = None,
) -> tuple[int, int]:
    valid_length = raw_num_samples
    chosen_crop_start = 0

    if max_samples is not None and raw_num_samples > max_samples:
        if crop_start is not None:
            start = min(max(crop_start, 0), raw_num_samples - max_samples)
        elif random_crop:
            start = random.randint(0, raw_num_samples - max_samples)
        else:
            start = max((raw_num_samples - max_samples) // 2, 0)
        chosen_crop_start = start
        valid_length = max_samples

    return valid_length, chosen_crop_start


def protocol_for_split(data_dir: Path, split: str) -> Path:
    try:
        protocol_name = DEFAULT_PROTOCOLS[split]
    except KeyError as exc:
        valid = ", ".join(sorted(DEFAULT_PROTOCOLS))
        raise ValueError(f"Unknown split {split!r}. Valid values: {valid}") from exc
    return data_dir / protocol_name


def audio_path_for_file_id(data_dir: Path, file_id: str) -> Path:
    if file_id.startswith("T_"):
        folder = "flac_T"
    elif file_id.startswith("D_"):
        folder = "flac_D"
    elif file_id.startswith("E_"):
        folder = "flac_E_eval"
    else:
        raise ValueError(f"Cannot infer audio folder for file id {file_id!r}")
    return data_dir / folder / f"{file_id}.flac"


def read_asvspoof5_protocol(
    data_dir: str | Path,
    split: str,
    *,
    protocol_path: str | Path | None = None,
    limit: int | None = None,
    limit_per_class: int | None = None,
) -> list[AudioItem]:
    data_root = Path(data_dir)
    protocol = Path(protocol_path) if protocol_path else protocol_for_split(data_root, split)
    if not protocol.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol}")

    items: list[AudioItem] = []
    class_counts: Counter[int] = Counter()

    with protocol.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            label_text = parts[-2].lower()
            if label_text not in LABEL_TO_ID:
                continue

            label = LABEL_TO_ID[label_text]
            if limit_per_class is not None and class_counts[label] >= limit_per_class:
                if all(class_counts.get(class_id, 0) >= limit_per_class for class_id in LABEL_TO_ID.values()):
                    break
                continue

            speaker_id = parts[0]
            file_id = parts[1]
            path = audio_path_for_file_id(data_root, file_id)
            if not path.exists():
                continue

            items.append(AudioItem(path=path, label=label, file_id=file_id, speaker_id=speaker_id))
            class_counts[label] += 1

            if limit is not None and len(items) >= limit:
                break

    if not items:
        raise ValueError(f"No labeled audio items were found in {protocol}")

    return items


def load_audio(
    path: str | Path,
    *,
    sample_rate: int,
    max_samples: int | None,
    random_crop: bool,
    crop_start: int | None = None,
    normalize: bool = True,
) -> AudioLoadResult:
    audio, current_sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).transpose(0, 1)

    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0).float()

    if current_sample_rate != sample_rate:
        waveform = AF.resample(waveform, current_sample_rate, sample_rate)

    if normalize and waveform.numel() > 0:
        waveform = waveform - waveform.mean()
        scale = waveform.abs().max().clamp_min(1e-5)
        waveform = waveform / scale

    original_length = waveform.numel()
    valid_length = original_length
    chosen_crop_start = 0

    if max_samples is not None:
        if waveform.numel() > max_samples:
            if crop_start is not None:
                start = min(max(crop_start, 0), waveform.numel() - max_samples)
            elif random_crop:
                start = random.randint(0, waveform.numel() - max_samples)
            else:
                start = max((waveform.numel() - max_samples) // 2, 0)
            waveform = waveform[start : start + max_samples]
            chosen_crop_start = start
            valid_length = max_samples
        elif waveform.numel() < max_samples:
            valid_length = waveform.numel()
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.numel()))

    return AudioLoadResult(
        waveform=waveform,
        valid_length=valid_length,
        crop_start=chosen_crop_start,
        original_length=original_length,
    )


def wav2vec_cache_path(cache_dir: str | Path, split: str, file_id: str) -> Path:
    cache_root = Path(cache_dir)
    shard = file_id[2:5]
    return cache_root / split / shard / f"{file_id}.pt"


def mfcc_cache_path(cache_dir: str | Path, split: str, file_id: str) -> Path:
    cache_root = Path(cache_dir)
    shard = file_id[2:5]
    return cache_root / split / shard / f"{file_id}.pt"


def crop_cached_frame_features(
    cached_features: torch.Tensor,
    *,
    raw_num_samples: int,
    crop_start: int,
    crop_num_samples: int,
) -> tuple[torch.Tensor, int]:
    feature_length = int(cached_features.size(0))
    if feature_length == 0 or raw_num_samples <= 0:
        return cached_features, feature_length

    if crop_num_samples >= raw_num_samples:
        return cached_features, feature_length

    start_ratio = crop_start / raw_num_samples
    end_ratio = min(crop_start + crop_num_samples, raw_num_samples) / raw_num_samples
    start_frame = min(int(start_ratio * feature_length), max(feature_length - 1, 0))
    end_frame = max(int(math.ceil(end_ratio * feature_length)), start_frame + 1)
    end_frame = min(end_frame, feature_length)
    sliced = cached_features[start_frame:end_frame]
    return sliced, int(sliced.size(0))


def crop_cached_ssl_features(
    cached_features: torch.Tensor,
    *,
    raw_num_samples: int,
    crop_start: int,
    crop_num_samples: int,
) -> tuple[torch.Tensor, int]:
    return crop_cached_frame_features(
        cached_features,
        raw_num_samples=raw_num_samples,
        crop_start=crop_start,
        crop_num_samples=crop_num_samples,
    )


def expected_mfcc_frames(num_samples: int, *, hop_length: int) -> int:
    if num_samples <= 0:
        return 1
    return max(int(num_samples // hop_length) + 1, 1)


class ASVspoof5Dataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        *,
        protocol_path: str | Path | None = None,
        sample_rate: int = 16_000,
        max_seconds: float = 4.0,
        random_crop: bool = False,
        limit: int | None = None,
        limit_per_class: int | None = None,
        ssl_cache_dir: str | Path | None = None,
        mfcc_cache_dir: str | Path | None = None,
        load_waveform: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate) if max_seconds > 0 else None
        self.random_crop = random_crop
        self.ssl_cache_dir = Path(ssl_cache_dir) if ssl_cache_dir else None
        self.mfcc_cache_dir = Path(mfcc_cache_dir) if mfcc_cache_dir else None
        self.load_waveform = load_waveform
        self.items = read_asvspoof5_protocol(
            self.data_dir,
            split,
            protocol_path=protocol_path,
            limit=limit,
            limit_per_class=limit_per_class,
        )

    def __len__(self) -> int:
        return len(self.items)

    @property
    def class_counts(self) -> Counter[int]:
        return Counter(item.label for item in self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        ssl_features = None
        ssl_length = None
        mfcc_features = None
        mfcc_length = None
        ssl_payload = None
        mfcc_payload = None
        raw_num_samples = None
        if self.ssl_cache_dir is not None:
            cache_path = wav2vec_cache_path(self.ssl_cache_dir, self.split, item.file_id)
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing wav2vec cache for {item.file_id}: {cache_path}")
            ssl_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            raw_num_samples = int(ssl_payload["raw_num_samples"])
        if self.mfcc_cache_dir is not None:
            cache_path = mfcc_cache_path(self.mfcc_cache_dir, self.split, item.file_id)
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing MFCC cache for {item.file_id}: {cache_path}")
            mfcc_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if raw_num_samples is None:
                raw_num_samples = int(mfcc_payload["raw_num_samples"])

        if self.load_waveform:
            audio = load_audio(
                item.path,
                sample_rate=self.sample_rate,
                max_samples=self.max_samples,
                random_crop=self.random_crop,
            )
            waveform = audio.waveform
            valid_length = audio.valid_length
            crop_start = audio.crop_start
        else:
            if raw_num_samples is None:
                raise ValueError("A cache with raw_num_samples metadata is required when load_waveform=False")
            valid_length, crop_start = compute_crop_window(
                raw_num_samples,
                max_samples=self.max_samples,
                random_crop=self.random_crop,
            )
            waveform = None

        if self.ssl_cache_dir is not None:
            assert ssl_payload is not None
            cached_features = ssl_payload["features"].float()
            current_raw_num_samples = int(ssl_payload["raw_num_samples"])
            if self.max_samples is None:
                ssl_features = cached_features
                ssl_length = int(cached_features.size(0))
            else:
                ssl_features, ssl_length = crop_cached_ssl_features(
                    cached_features,
                    raw_num_samples=current_raw_num_samples,
                    crop_start=crop_start,
                    crop_num_samples=min(self.max_samples, current_raw_num_samples),
                )
        if self.mfcc_cache_dir is not None:
            assert mfcc_payload is not None
            cached_features = mfcc_payload["features"].float()
            current_raw_num_samples = int(mfcc_payload["raw_num_samples"])
            hop_length = int(mfcc_payload.get("hop_length", int(0.010 * self.sample_rate)))
            if self.max_samples is None:
                mfcc_features = cached_features
                mfcc_length = int(cached_features.size(0))
            else:
                mfcc_features, mfcc_length = crop_cached_frame_features(
                    cached_features,
                    raw_num_samples=current_raw_num_samples,
                    crop_start=crop_start,
                    crop_num_samples=min(self.max_samples, current_raw_num_samples),
                )
                target_frames = expected_mfcc_frames(self.max_samples, hop_length=hop_length)
                current_frames = int(mfcc_features.size(0))
                if current_frames < target_frames:
                    mfcc_features = torch.nn.functional.pad(mfcc_features, (0, 0, 0, target_frames - current_frames))
                elif current_frames > target_frames:
                    mfcc_features = mfcc_features[:target_frames]
                mfcc_length = target_frames
        return {
            "waveform": waveform,
            "length": valid_length,
            "label": item.label,
            "path": str(item.path),
            "file_id": item.file_id,
            "ssl_features": ssl_features,
            "ssl_length": ssl_length,
            "mfcc_features": mfcc_features,
            "mfcc_length": mfcc_length,
        }


def collate_audio(batch: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(batch)
    waveforms = None
    if all(row.get("waveform") is not None for row in rows):
        waveforms = pad_sequence([row["waveform"] for row in rows], batch_first=True)
    ssl_features = None
    ssl_lengths = None
    mfcc_features = None
    mfcc_lengths = None
    if all(row.get("ssl_features") is not None for row in rows):
        ssl_features = pad_sequence([row["ssl_features"] for row in rows], batch_first=True)
        ssl_lengths = torch.tensor([row["ssl_length"] for row in rows], dtype=torch.long)
    if all(row.get("mfcc_features") is not None for row in rows):
        mfcc_features = pad_sequence([row["mfcc_features"] for row in rows], batch_first=True)
        mfcc_lengths = torch.tensor([row["mfcc_length"] for row in rows], dtype=torch.long)
    return {
        "waveforms": waveforms,
        "lengths": torch.tensor([row["length"] for row in rows], dtype=torch.long),
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        "paths": [row["path"] for row in rows],
        "file_ids": [row["file_id"] for row in rows],
        "ssl_features": ssl_features,
        "ssl_lengths": ssl_lengths,
        "mfcc_features": mfcc_features,
        "mfcc_lengths": mfcc_lengths,
    }


def make_balanced_sampler(dataset: ASVspoof5Dataset) -> WeightedRandomSampler:
    counts = dataset.class_counts
    weights = [1.0 / counts[item.label] for item in dataset.items]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
