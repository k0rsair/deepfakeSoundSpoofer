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

You can also train only one branch with `--model-type wav2vec_pyara`, `--model-type wav2vec_temporal`, or `--model-type mfcc_resnet`.
The temporal wav2vec head keeps channel mixing pointwise and models only the time axis, which is often a cleaner fit for SSL embeddings than treating them like a 2D map.

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

If you have precomputed MFCC features, add `--mfcc-cache-dir cache\mfcc_sr16k_40` to skip MFCC extraction during training and evaluation.

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

Train only wav2vec with the temporal head:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type wav2vec_temporal `
  --bundle WAV2VEC2_XLSR_300M `
  --freeze-transformer-layers 8 `
  --epochs 5 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --max-seconds 3 `
  --output-dir runs/xlsr300m_temporal
```

Train fusion with the temporal wav2vec head plus MFCC/ResNet:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type fusion_temporal `
  --bundle WAV2VEC2_XLSR_300M `
  --freeze-wav2vec `
  --wav2vec-layers 21 `
  --wav2vec-cache-dir cache\xlsr300m_l21 `
  --epochs 20 `
  --batch-size 16 `
  --max-seconds 4 `
  --output-dir runs\fusion_temporal_l21_cache
```

## Resume training

`last.pt` stores the latest completed epoch together with optimizer and scheduler state, so a stopped run can be continued with:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --resume-from runs\fusion_frozen_l21_cache\last.pt
```

`--resume-from` reuses the saved training config by default. Pass a larger `--epochs` if you want to extend the total budget, for example `--epochs 30`. Use `last.pt` for resuming and `best.pt` for evaluation or inference.

## Evaluate

Run the neural checkpoint on a labeled split and print `loss`, `accuracy`, and `roc_auc` as JSON:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.evaluate `
  --checkpoint runs\fusion_frozen_l21_cache\best.pt `
  --split dev `
  --batch-size 16
```

If you want to save the report, add `--output runs\fusion_frozen_l21_cache\dev_metrics.json`.
The report now also includes `precision`, `recall`, `f1`, confusion matrix, `average_precision`, and `eer`.
If you want per-file scores, add `--predictions-csv runs\fusion_frozen_l21_cache\eval_predictions.csv` or `--predictions-jsonl ...`.
If you want plots, add `--plots-dir runs\fusion_frozen_l21_cache\eval_plots`.
If the checkpoint was trained with a wav2vec cache but the selected split has no cached features, evaluation automatically falls back to running wav2vec directly.

## Interpret

Build an interpretation report from saved per-file predictions and the matching ASVspoof5 protocol:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.interpret `
  --predictions-csv runs\fusion_temporal_l21_balanced\eval_predictions.csv `
  --protocol data\ASVspoof5.eval.track_1.tsv `
  --output-dir runs\fusion_temporal_l21_balanced\interpret `
  --threshold 0.5
```

The report writes `interpret_summary.json`, grouped error breakdowns, threshold sweep tables, top false positives, top false negatives, borderline cases, and plots. For fast threshold iterations without PNG rendering, add `--no-plots`.
Use `dev` predictions for threshold selection, then apply the selected threshold once to `eval`.

## Explain

Explain a single model decision with temporal occlusion and branch ablation:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.explain `
  data\flac_E_eval\E_0000686352.flac `
  --checkpoint runs\fusion_temporal_l21_balanced\best.pt `
  --device cuda `
  --window-seconds 0.8 `
  --hop-seconds 0.4 `
  --output-json runs\fusion_temporal_l21_balanced\explain_E_0000686352\explanation.json `
  --output-csv runs\fusion_temporal_l21_balanced\explain_E_0000686352\occlusion.csv `
  --plots-dir runs\fusion_temporal_l21_balanced\explain_E_0000686352\plots
```

The JSON contains the base prediction, the time windows that most increase the spoof decision, and for fusion checkpoints the decision after disabling the wav2vec or MFCC branch.

For a more formal LIME-style local surrogate, add random segment masking:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.explain `
  data\flac_E_eval\E_0000686352.flac `
  --checkpoint runs\fusion_temporal_l21_balanced\best.pt `
  --device cuda `
  --window-seconds 0.8 `
  --hop-seconds 0.4 `
  --lime-samples 400 `
  --lime-segments 20 `
  --lime-output-csv runs\fusion_temporal_l21_balanced\explain_E_0000686352_lime\lime_segments.csv `
  --output-json runs\fusion_temporal_l21_balanced\explain_E_0000686352_lime\explanation.json `
  --plots-dir runs\fusion_temporal_l21_balanced\explain_E_0000686352_lime\plots
```

This fits a weighted Ridge surrogate over binary segment masks and reports `weighted_r2`, which should be treated as the local explanation fidelity.

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

## Web Demo

Run a lightweight browser demo for uploads and live inference:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.web_demo `
  --checkpoint runs\fusion_frozen_l21_cache\best.pt `
  --device cuda `
  --host 127.0.0.1 `
  --port 7860
```

Then open `http://127.0.0.1:7860` in your browser, upload an audio file, and the page will show the predicted label plus `bonafide/spoof` probabilities.
The web interface is in Russian by default.

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

## MFCC cache

If you want a faster MFCC-only or fusion run, you can precompute the MFCC matrices once:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.cache_mfcc `
  --data-dir data `
  --output-dir cache\mfcc_sr16k_40 `
  --splits train dev `
  --batch-size 32 `
  --dtype float16
```

Then train the MFCC-only model from cache:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.train `
  --model-type mfcc_resnet `
  --mfcc-cache-dir cache\mfcc_sr16k_40 `
  --epochs 20 `
  --batch-size 16 `
  --max-seconds 4 `
  --output-dir runs\mfcc_resnet_cache
```

Evaluation can also reuse the same cache:

```powershell
.\.venv\Scripts\python.exe -m deepfake_spoofer.evaluate `
  --checkpoint runs\mfcc_resnet_cache\best.pt `
  --split eval `
  --mfcc-cache-dir cache\mfcc_sr16k_40
```
