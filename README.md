# DeepfakeSpoofer

Fusion detector for ASVspoof5 audio deepfake detection.

The default model combines two PyAra-inspired vectors:

```text
raw audio
  -> wav2vec/XLSR + PyAra/AASIST graph head -> 160-d vector
  -> MFCC + ResNet head                    -> 128-d vector
  -> concat                                -> 288-d fusion vector
  -> neural classifier and optional KNN head
```

You can also train only one branch with `--model-type wav2vec_pyara` or `--model-type mfcc_resnet`.

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

This trains only a tiny balanced subset and is meant to check that data loading, CUDA, wav2vec, MFCC/ResNet, fusion and checkpointing all work.

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type fusion `
  --bundle WAV2VEC2_BASE `
  --freeze-wav2vec `
  --wav2vec-layers 1 `
  --limit-per-class 2 `
  --epochs 1 `
  --batch-size 2 `
  --max-seconds 1 `
  --output-dir runs/smoke_fusion
```

## Full training

Fine-tune part of XLSR while keeping the convolutional feature extractor and the earliest transformer layers frozen. This is a safer first run for a 12 GB GPU:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type fusion `
  --bundle WAV2VEC2_XLSR_300M `
  --epochs 5 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --max-seconds 3 `
  --lr-head 1e-4 `
  --lr-wav2vec 1e-5 `
  --wav2vec-layers 12 `
  --freeze-transformer-layers 8 `
  --output-dir runs/xlsr300m_fusion
```

For a full 24-layer XLSR fine-tune, omit `--wav2vec-layers`; if VRAM is tight, keep `--batch-size 1` and raise `--grad-accum-steps`.

Train only the PyAra/AASIST-style head and keep wav2vec fully frozen:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type fusion `
  --bundle WAV2VEC2_XLSR_300M `
  --freeze-wav2vec `
  --epochs 5 `
  --batch-size 8 `
  --max-seconds 4 `
  --output-dir runs/xlsr300m_fusion_frozen
```

Use only the first N wav2vec transformer layers:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train --wav2vec-layers 12
```

Train only MFCC/ResNet:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type mfcc_resnet `
  --epochs 5 `
  --batch-size 32 `
  --max-seconds 4 `
  --output-dir runs/mfcc_resnet
```

Train only wav2vec/PyAra:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type wav2vec_pyara `
  --bundle WAV2VEC2_XLSR_300M `
  --freeze-transformer-layers 8 `
  --epochs 5 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --max-seconds 3 `
  --output-dir runs/xlsr300m_pyara
```

## KNN head

Fit KNN on the fusion vectors from the train split:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.knn fit `
  --checkpoint runs\xlsr300m_fusion\best.pt `
  --split train `
  --batch-size 8 `
  --neighbors 10 `
  --output runs\xlsr300m_fusion\knn.pkl
```

Evaluate neural, KNN and averaged probabilities on dev or eval:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.knn eval `
  --checkpoint runs\xlsr300m_fusion\best.pt `
  --knn runs\xlsr300m_fusion\knn.pkl `
  --split dev `
  --batch-size 8
```

## Predict

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.predict `
  --checkpoint runs\xlsr300m_fusion\best.pt `
  data\flac_D\D_0000000001.flac
```

Predict with neural + KNN heads:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.knn predict `
  --checkpoint runs\xlsr300m_fusion\best.pt `
  --knn runs\xlsr300m_fusion\knn.pkl `
  data\flac_D\D_0000000001.flac
```

## Frozen wav2vec cache

When `--freeze-wav2vec` is enabled, wav2vec does the same work every epoch. You can precompute one chosen layer once and train from cache afterward.

The cache script stores the exact layer selected by `--wav2vec-layers`. A practical starting point is layer `21`:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.cache_wav2vec `
  --data-dir data `
  --output-dir cache\xlsr300m_l21 `
  --bundle WAV2VEC2_XLSR_300M `
  --wav2vec-layers 21 `
  --splits train dev `
  --batch-size 8 `
  --dtype float16
```

Then train with frozen wav2vec directly from cache:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type fusion `
  --bundle WAV2VEC2_XLSR_300M `
  --freeze-wav2vec `
  --wav2vec-layers 21 `
  --wav2vec-cache-dir cache\xlsr300m_l21 `
  --epochs 20 `
  --batch-size 16 `
  --grad-accum-steps 1 `
  --max-seconds 4 `
  --lr-head 5e-5 `
  --weight-decay 5e-4 `
  --no-balanced-sampler `
  --class-weights `
  --label-smoothing 0.05 `
  --scheduler cosine `
  --warmup-ratio 0.1 `
  --eval-passes 1 `
  --early-stopping-patience 5 `
  --early-stopping-min-epochs 4 `
  --output-dir runs\fusion_frozen_l21_cache
```

Notes:

- The train command validates that cache bundle and cache layer match the requested `--bundle` and `--wav2vec-layers`.
- `eval` is much larger than `train` and `dev`; cache it only when you know you need it.
- Full-sequence wav2vec caches can be large on disk, especially for `train + dev + eval`.
