from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torchaudio.functional as AF
import soundfile as sf
from torch.utils.data import Dataset, WeightedRandomSampler


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
    normalize: bool = True,
) -> tuple[torch.Tensor, int]:
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

    valid_length = waveform.numel()

    if max_samples is not None:
        if waveform.numel() > max_samples:
            if random_crop:
                start = random.randint(0, waveform.numel() - max_samples)
            else:
                start = max((waveform.numel() - max_samples) // 2, 0)
            waveform = waveform[start : start + max_samples]
            valid_length = max_samples
        elif waveform.numel() < max_samples:
            valid_length = waveform.numel()
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.numel()))

    return waveform, valid_length


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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate) if max_seconds > 0 else None
        self.random_crop = random_crop
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
        waveform, length = load_audio(
            item.path,
            sample_rate=self.sample_rate,
            max_samples=self.max_samples,
            random_crop=self.random_crop,
        )
        return {
            "waveform": waveform,
            "length": length,
            "label": item.label,
            "path": str(item.path),
            "file_id": item.file_id,
        }


def collate_audio(batch: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(batch)
    return {
        "waveforms": torch.stack([row["waveform"] for row in rows]),
        "lengths": torch.tensor([row["length"] for row in rows], dtype=torch.long),
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        "paths": [row["path"] for row in rows],
        "file_ids": [row["file_id"] for row in rows],
    }


def make_balanced_sampler(dataset: ASVspoof5Dataset) -> WeightedRandomSampler:
    counts = dataset.class_counts
    weights = [1.0 / counts[item.label] for item in dataset.items]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
