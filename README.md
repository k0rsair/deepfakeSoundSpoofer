# DeepfakeSpoofer

Wav2vec + PyAra/AASIST-style detector for ASVspoof5 audio deepfake detection.

The model follows the idea from `efanov/pyara`: extract SSL features with wav2vec/XLSR, project them, then run a residual attention + graph pooling head for `bonafide` vs `spoof` classification.

## Data layout

The code expects this structure:

```text
data/
  ASVspoof5.train.tsv
  ASVspoof5.dev.track_1.tsv
  ASVspoof5.eval.track_1.tsv
  flac_T/
  flac_D/
  flac_E_eval/
```

Labels are read from the second-to-last TSV column: `bonafide` -> `0`, `spoof` -> `1`.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The current environment was verified with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Quick smoke run

This trains only a tiny balanced subset and is meant to check that data loading, CUDA, wav2vec and the head all work.

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --bundle WAV2VEC2_BASE `
  --freeze-wav2vec `
  --wav2vec-layers 1 `
  --limit-per-class 2 `
  --epochs 1 `
  --batch-size 2 `
  --max-seconds 1 `
  --output-dir runs/smoke
```

## Full training

Fine-tune part of XLSR while keeping the convolutional feature extractor and the earliest transformer layers frozen. This is a safer first run for a 12 GB GPU:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --bundle WAV2VEC2_XLSR_300M `
  --epochs 5 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --max-seconds 3 `
  --lr-head 1e-4 `
  --lr-wav2vec 1e-5 `
  --wav2vec-layers 12 `
  --freeze-transformer-layers 8 `
  --output-dir runs/xlsr300m_pyara
```

For a full 24-layer XLSR fine-tune, omit `--wav2vec-layers`; if VRAM is tight, keep `--batch-size 1` and raise `--grad-accum-steps`.

Train only the PyAra/AASIST-style head and keep wav2vec fully frozen:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --bundle WAV2VEC2_XLSR_300M `
  --freeze-wav2vec `
  --epochs 5 `
  --batch-size 8 `
  --max-seconds 4 `
  --output-dir runs/xlsr300m_frozen
```

Use only the first N wav2vec transformer layers:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train --wav2vec-layers 12
```

## Predict

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.predict `
  --checkpoint runs/xlsr300m_pyara/best.pt `
  data\flac_D\D_0000000001.flac
```
