# Dubai aerial segmentation: baseline U-Net vs. dilated bottleneck

Assignment B – a reproducible segmentation experiment (Advanced Computer Vision, Unit 3).

One question: **does replacing the U-Net bottleneck with parallel multi-rate dilated convolutions change
test segmentation quality, and at what parameter / latency cost?** Everything except the bottleneck is held fixed.

Every number in this README comes from the run whose outputs are in `logs/` and `results/`
(training on 14 Nov 2026, NVIDIA GeForce RTX 4050 Laptop GPU). Nothing is hand-entered into the figures:
they are drawn by `evaluate.py` from the checkpoints, `logs/training_log.csv` and `results/metrics_summary.json`.

## Repository layout

| Path | Content |
|---|---|
| `dataset.py` | Data contract: download, pairing, class/ignore mapping, split, transforms; writes `data_contract.json` |
| `models.py` | `SegmentationUNet` (baseline / dilated bottleneck), parameter table, conv-MAC counter |
| `train.py` | Seeded training of both models, early stopping, per-epoch log; writes `experiment_config.json`, `environment.txt` |
| `evaluate.py` | **The exact evaluation script**: test metrics, latency, figures, dashboard |
| `data_contract.json` | Class IDs, ignore rule, transforms, exact file list per split, pixel counts per split |
| `experiment_config.json` | All hyper-parameters, the stopping rule, parameter table and training outcome |
| `environment.txt`, `requirements.txt` | Python 3.10.0, torch 2.5.1+cu121, torchvision 0.20.1, CUDA 12.1, cuDNN 9.1, driver 592.82 |
| `logs/training_log.csv` | One row per epoch per model: `model, epoch, lr, train_loss, val_loss, val_miou, epoch_time_s` |
| `logs/train_stdout.txt`, `logs/evaluate_stdout.txt` | Console output of the reported runs |
| `logs/smoke_training_log.csv` | 1-epoch smoke test (not used for any result) |
| `results/metrics_summary.json` | Per-class IoU, mIoU, pixel accuracy, params, MACs, latency + conditions, confusion matrices, per-image mIoU, test image names |
| `results/per_class_iou.csv`, `results/per_image_miou.csv`, `results/param_table.txt` | Tables |
| `results/*.png` | `results_dashboard`, `training_curves`, `qualitative_comparison`, `failure_cases` |

## Reproduction

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python dataset.py     # downloads via kagglehub into data/, writes data_contract.json
python models.py      # block-by-block parameter table + MACs
python train.py --smoke   # optional 1-epoch smoke test (separate log/checkpoints)
python train.py       # baseline, then dilated -> checkpoints/, logs/training_log.csv, experiment_config.json
python evaluate.py    # test split once -> results/
```

`data/` and `checkpoints/` are not part of the submission (`.gitignore`). The full run took about 35 s (baseline)
and 69 s (dilated) of training on the laptop GPU, plus data loading.

---

## Task 1 – Data contract

* **Dataset:** *Semantic segmentation of aerial imagery*, Humans in the Loop / MBRSC, Dubai, CC0.
  `kagglehub.dataset_download("humansintheloop/semantic-segmentation-of-aerial-imagery")`.
  72 image/mask pairs in 8 tiles (9 per tile). File names repeat across tiles, so pairs are matched **within** a
  tile and named `TileN_part_XXX`.
* **Reference labels:** the human-annotated RGB masks shipped with the dataset. No pseudo-ground truth is used anywhere.
* **Classes** (RGB as present in the PNGs):

  | ID | Class | RGB |
  |---|---|---|
  | 0 | Building | (60, 16, 152) |
  | 1 | Land (unpaved) | (132, 41, 246) |
  | 2 | Road | (110, 193, 228) |
  | 3 | Vegetation | (254, 221, 58) |
  | 4 | Water | (226, 169, 41) |
  | 255 | ignore (Unlabeled (155,155,155) and anything off-palette) | – |

* **Ignore rule:** each mask pixel goes to the nearest palette colour (L1 over 5 classes + Unlabeled). If the
  nearest is Unlabeled, or the distance is ≥ 45, the label is 255. It is excluded from the loss
  (`ignore_index=255`) and from every confusion matrix.
  Inspection of all 72 masks found exactly 7 colours: the 5 classes, Unlabeled and pure black (0,0,0)
  (~0.5 % of pixels, at tile borders). Black is ≥ 228 L1 from every palette entry, so it becomes 255.
  18 masks are palette-mode PNGs and are converted to RGB first. The dataset's `classes.json` lists different hex
  colours from those actually in the masks; the contract follows the masks.
* **Split:** pairs sorted by (tile, file), shuffled with `random.Random(42)`, cut 70/15/15 → **50 / 10 / 12**.
  The exact file list of every split is in `data_contract.json`. The 12 test images come from Tiles 1, 2, 4 and 7.
* **Transforms:** image → PIL bilinear resize 256×256 → `to_tensor` → ImageNet normalisation.
  Mask → PIL nearest resize 256×256 → index mapping. Nearest resizing only samples original pixels, so this
  equals mapping first and then resizing.
  **Train only:** synchronised hflip p=0.5, vflip p=0.5, rotation by 90/180/270° p=0.5. Done with exact
  `flip`/`rot90` pixel permutations, so the mask needs no interpolation.
* **Test-set GT pixels (256×256):** Building 126,105 (16.3 %), Land 422,821 (54.6 %), Road 102,733 (13.3 %),
  Vegetation 41,843 (5.4 %), Water 81,060 (10.5 %), ignored 11,870.

## Task 2 – Baseline

* **Architecture:** 4-stage U-Net, `base_ch=32`. Encoder DoubleConv stages 32-64-128-256
  (DoubleConv = 2 × [Conv3×3 no bias + BN + ReLU]) with MaxPool 2×2 between them.
  Bottleneck `DoubleConv(256, 512)` at 1/16 resolution (16×16). Decoder: 4 × [ConvTranspose2d 2×2 s2 → concat skip
  → DoubleConv]. 1×1 head to 5 classes. **7,763,173 parameters.**
* **Seed / determinism:** seed 42 for `random`, `numpy` and `torch`; `cudnn.deterministic=True`,
  `cudnn.benchmark=False`. Batch order uses a `torch.Generator` seeded 42. Augmentation uses its own
  `random.Random(42)`. Both are re-created for each model, so **both models see the same batches with the same
  augmentations**.
* **Resolution:** 256×256. **Augmentation:** as in Task 1.
* **Optimisation:** `CrossEntropyLoss(ignore_index=255)` (unweighted), AdamW (lr 1e-3, wd 1e-4),
  CosineAnnealingLR (T_max 60, eta_min 1e-5, stepped per epoch), batch 8, max 60 epochs.
* **Stopping rule:** stop after 15 consecutive epochs without validation-mIoU improvement. Keep the
  best-val-mIoU checkpoint. The test set was never used for any decision.
* **Outcome:** best val mIoU **0.5414 at epoch 15**, early-stopped at epoch 30.

## Task 3 – Controlled change: dilation in the bottleneck

Only `model.bottleneck` changes. Encoder, decoder, skips, head, data, augmentation stream, optimiser, schedule and
stopping rule are the same code with the same settings.

`ParallelDilatedBottleneck`: ConvBNReLU 3×3 stem 256→512, then 4 parallel ConvBNReLU 3×3 branches 512→128 with
dilation = padding = 1, 2, 4, 8, concatenated to 512, then a 1×1 ConvBNReLU projection 512→512.
At the 16×16 bottleneck, rate 8 gives a 17×17 effective kernel, covering the whole feature map.

**Comparison budget** (`python models.py`, `results/param_table.txt`):

| block | baseline | dilated | delta |
|---|---:|---:|---:|
| enc1–enc4 | 1,173,216 | 1,173,216 | 0 |
| bottleneck | 3,540,992 | 3,804,160 | **+263,168** |
| up4–up1 | 3,048,800 | 3,048,800 | 0 |
| head | 165 | 165 | 0 |
| **total** | **7,763,173** | **8,026,341** | **+263,168 (+3.39 %)** |
| conv MACs @256² | 12.080 G | 12.147 G | +67.1 M (+0.56 %) |

The stem matches the baseline's first bottleneck conv (256·512·9 weights). The four branches together have exactly
the weights of the baseline's second conv (4·512·128·9 = 512·512·9). **The whole parameter difference is the 1×1
projection** (512·512 + 1,024 BN = 263,168), and the whole MAC difference too (512·512·16·16 = 67.1 M).

## Task 4 – Evidence

### Test results (12 images, 256×256, human reference masks)

| Class | GT pixels | IoU baseline | IoU dilated | Δ |
|---|---:|---:|---:|---:|
| Building | 126,105 | 0.4925 | 0.5089 | +0.0163 |
| Land | 422,821 | 0.6840 | 0.6899 | +0.0059 |
| Road | 102,733 | 0.2630 | 0.3477 | +0.0846 |
| Vegetation | 41,843 | 0.1954 | 0.1946 | −0.0008 |
| Water | 81,060 | 0.7128 | 0.7977 | +0.0849 |
| **mIoU** | | **0.4695** | **0.5077** | **+0.0382** |
| Pixel accuracy | | 0.7112 | 0.7261 | +0.0149 |
| Best val mIoU (epoch) | | 0.5414 (15) | 0.5672 (41) | |
| Epochs run (early stop) | | 30 | 56 | |
| Parameters | | 7,763,173 | 8,026,341 | +263,168 |

IoU comes from one confusion matrix accumulated over all test pixels (GT = 255 excluded). mIoU is the mean over the
5 classes. Confusion matrices and per-image mIoU are in `results/metrics_summary.json`. The dilated model has higher
per-image mIoU on 10 of 12 test images (`results/per_image_miou.csv`).

What the confusion matrices show (rows = GT):
* **Road:** recall rises from 33.8 % to 46.3 % (34,687 → 47,526 correct pixels). Road→Building errors drop from
  14,779 to 6,001. Road→Land (~36k) stays the largest road error for both models.
* **Water:** the gain comes from fewer false positives (pixels predicted as Water: 98,012 → 80,066), especially
  Land→Water and Road→Water. Water recall is in fact slightly lower (74,521 → 71,495).
* **Vegetation** stays poor for both models (precision ≈ 24–26 %). Most false vegetation is GT Land
  (31,241 / 37,175 pixels).

### Latency

| | baseline | dilated |
|---|---:|---:|
| **median of 5 per-block medians** | **12.51 ms** | **15.27 ms** |
| per-block medians (blocks alternated) | 12.98, 12.92, 12.51, 9.55, 11.89 | 15.51, 15.86, 15.27, 4.78, 12.00 |
| pooled mean ± sd (1,000 passes) | 12.67 ± 4.02 ms | 13.40 ± 5.34 ms |
| pooled p95 / min | 18.73 / 4.04 ms | 20.86 / 4.01 ms |

**Conditions:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, driver 592.82), torch 2.5.1+cu121, CUDA 12.1,
cudnn.benchmark = False, deterministic = True. Batch 1, input 1×3×256×256, FP32, `model.eval()` +
`torch.no_grad()`. Each block = 50 warm-up + 200 timed forward passes, `time.perf_counter` around each pass with
`torch.cuda.synchronize()`. 5 blocks per model, alternating baseline / dilated. Forward pass only: no data
loading, preprocessing or argmax.

**How to read it:** latency on this laptop GPU is very unstable (GPU boost / power state under Windows).
The dilated model was slower in 4 of the 5 paired blocks, but in block 4 it measured 4.78 ms against the
baseline's 9.55 ms. Earlier executions of the same script, with a single 50 + 200 block per model, gave medians of
11.31 / 12.71, 10.03 / 13.80 and 10.11 / 20.85 ms (baseline / dilated). Those differ only in plotting code and in
not repeating the latency block; test metrics were identical each time. A reasonable reading is that the dilated
bottleneck adds a few ms per forward pass at batch 1, though that size isn't pinned down. Its MAC overhead is only
0.56 %, so any real slowdown probably comes from launching 4 more small kernels plus a concat and the extra 1×1
layer at 16×16, and from dilated-convolution kernel efficiency, rather than from arithmetic. That explanation is a
hypothesis; it was not profiled.

### Figures

* `results/results_dashboard.png`: per-class IoU, per-class Δ, validation-mIoU curves, summary table.
* `results/training_curves.png`: train loss, val loss, val mIoU per epoch, from `logs/training_log.csv`.
* `results/qualitative_comparison.png`: the 3 test images with the largest per-image mIoU gain
  (Tile2_part_006 +0.136, Tile1_part_002 +0.097, Tile7_part_001 +0.069). Columns: source | reference mask |
  baseline | dilated | error map of the dilated model (green correct, red wrong, grey ignored).
* `results/failure_cases.png`: the 3 lowest test images by mean per-image mIoU of the two models
  (Tile1_part_004, Tile4_part_005, Tile4_part_009). Tile1_part_004 is mostly bright bare land with thin tracks.
  Both models nearly miss the narrow roads, and per-image mIoU is pulled down by small Building/Water regions.
  The Tile 4 images (circular residential clusters) show large Building/Vegetation/Land confusion around the
  clusters.

---

## What this experiment does NOT establish

1. **No statistical evidence.** One seed, one split, one training run per model, and no significance test or
   confidence interval. A +0.038 mIoU difference can be within seed-to-seed variance for a 50-image training set;
   repeated seeds and splits would be needed to tell.
2. **Unequal effective training length (confound).** With the same early-stopping rule, the baseline stopped at
   epoch 30 (lr still 5.3e-4) while the dilated model ran to epoch 56 and got its best checkpoint at epoch 41
   (lr 2.6e-4), in the low-learning-rate part of the cosine schedule the baseline never reached. Within the first
   30 epochs the dilated model's best val mIoU was 0.5483 vs 0.5414 for the baseline. Part of the reported gap may
   come from the stopping rule interacting with the schedule, not from dilation.
3. **Tiny test set.** 12 images (≈775k labelled pixels at 256²), and validation is only 10 images, which drives
   both model selection and early stopping.
4. **Spatial leakage and no generalisation claim.** The split is random over images, not tiles. Train, val and test
   images come from the same tiles (neighbouring parts of the same scenes), so the test score is optimistic. It says
   nothing about unseen tiles, other cities, sensors or seasons. A tile-held-out split would be needed for that.
5. **The change is not a perfectly isolated "dilation" effect.** The variant adds a 1×1 projection
   (+263,168 parameters, +3.4 %; +67 M MACs) and a stem + 4-branch structure. Its effect cannot be separated from
   multi-branch aggregation or extra capacity. The rates (1, 2, 4, 8) were not ablated, and there is no
   equal-parameter control (e.g. baseline + 1×1 layer, or all branches at rate 1).
6. **No per-model hyper-parameter tuning.** Both models use the same lr, weight decay, schedule and batch size,
   chosen a priori. Either model might do better with its own settings.
7. **Resized evaluation.** All metrics are computed at 256×256 after bilinear (image) and nearest (mask) resizing of
   tiles up to ~2,000 px wide, not at native resolution. Thin structures (roads) are degraded by the resizing itself.
8. **Rare classes rest on few pixels.** Vegetation has 41,843 test pixels (5.4 %), and per-class IoU on such classes
   is sensitive to a handful of regions or images.
9. **Label noise.** The reference masks are human polygon annotations with coarse boundaries and some
   inconsistency (e.g. ignored/Unlabeled regions, generalised building outlines). IoU measures agreement with these
   labels, not with the true scene.
10. **Narrow latency evidence.** Latency was measured on one laptop GPU, batch 1, FP32, forward pass only, with
    large run-to-run variance (see above). It does not cover CPU, other GPUs, larger batches, mixed precision,
    TensorRT/compiled models, end-to-end pipeline time, memory use or energy.
11. **Bitwise reproducibility not verified.** Seeds and cuDNN flags are fixed, but
    `torch.use_deterministic_algorithms` was not enabled, and some CUDA kernels (e.g. the backward pass of
    transposed convolutions and of the ignore-index loss) may be nondeterministic. A re-run may give slightly
    different numbers, and run-to-run training variance was not measured. Test inference was re-executed
    several times and gave identical metrics.
