# 4ind best-validation benchmark

This benchmark keeps the original `simplenet.py`, `simplenet_plus.py`, and
`simplenet_gl.py` files unchanged.  It evaluates two products independently
with one shared, portable split manifest.

## Fixed protocol

- Products: `KQG27542`, `KQG27824`
- Excluded block: `20241222_20_결함_2`
- Excluded labels: `라벨링불가`, `라벨링 불가`, `판단불가`, `멈춤`
- Train pool: normal images from 2024-12-21 through 2024-12-23
- Train sample: deterministic 10% with seed 0, stratified by source block and
  spread across contiguous temporal bins inside each block
- Validation: all usable images from 2024-12-24 through 2024-12-25
- Test: all usable images from 2024-12-26 through 2024-12-27
- Selection: highest validation I-AUROC; exact ties keep the earlier epoch
- Test access: once, after the selected checkpoint is restored and verified
- Pixel metrics: unavailable because the supplied archives contain no masks

The complete extracted inventory is checked before training.  With the
excluded block removed, the manifest generator expects 146,840 images.

| Product | Train pool | Selected train | Validation | Test |
|---|---:|---:|---:|---:|
| KQG27542 | 19,302 | 1,930 | 1,384 | 6,463 |
| KQG27824 | 24,005 | 2,400 | 529 | 3,358 |

## Server command

The launcher accepts the dataset root at runtime, so the local and server
absolute paths do not need to match.  Paths written to the manifest are
relative to this root.

```bash
./run_fourind_bestval_multigpu.sh \
  /server/path/to/4ind_dataset_202608 \
  /server/path/to/results_fourind_bestval \
  0,1,2
```

The dataset root may instead be supplied through `FOURIND_DATA_ROOT`:

```bash
FOURIND_DATA_ROOT=/server/path/to/4ind_dataset_202608 \
  ./run_fourind_bestval_multigpu.sh "" /server/path/to/results 0,1,2
```

### Date-mixed re-experiment

Set `FOURIND_SPLIT_STRATEGY=date-mixed` to retain the original per-product
train/validation/test counts and the exact validation/test count for each
normal/defect label, while redistributing every split across the available
capture dates and source blocks.  Model, augmentation, input, training, and
best-validation settings are unchanged.

```bash
FOURIND_SPLIT_STRATEGY=date-mixed \
  ./run_fourind_bestval_multigpu.sh \
  /workspace/4ind \
  /workspace/SimpleNet/4ind_result_date_mixed \
  2,3,4
```

Use a new results directory.  The launcher records the strategy in
`fourind_manifest_date_mixed_seed0.summary.json`, including per-split date and
source-block counts.  The launcher uses two explicit steps: all three methods
for KQG27542 run together, followed by all three methods for KQG27824.  GPU
assignments remain fixed by method across both steps.

Exactly three GPU ids are required.  A completed `job_result.json` is skipped,
while an unfinished job resumes from `checkpoint_progress.pth`.

## Methods and inputs

| Method | Training augmentation | Input policy |
|---|---:|---|
| SimpleNet | 0.0 | complete image letterboxed to 320x320 |
| SimpleNet++ | 0.1 | complete image letterboxed to 320x320 |
| Global/local (Ours) | 0.1 | 128x2048 global view plus 512x512 local tiles, stride 384 |

All methods use WRN50 layers 2/3, batch size 8, 40 meta epochs, four GAN
epochs, and seed 0.  Each product/method job performs training, validation,
selected-checkpoint verification, and final test evaluation on the same GPU.

## Outputs

```text
results_root/
├── fourind_manifest_<strategy>_seed0.csv
├── fourind_manifest_<strategy>_seed0.summary.json
├── simplenet/<product>/
├── simplenet_plus/<product>/
├── global_local/<product>/
├── results.csv
└── summary.json
```

Each job stores all validation epochs, the best-validation checkpoint, final
test I-AUROC, per-defect I-AUROC, and model/end-to-end inference time.  The
merge step checks six-job completeness, manifest identity, epoch history,
selection policy, checkpoint re-evaluation, finite AUROC values, and the
expected split sizes.

## PoC note

`--smoke` limits a job to four train images and two normal plus two anomaly
images in each evaluation split.  Smoke results validate the code path only;
they are not estimates of benchmark performance.
