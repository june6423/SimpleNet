# Global-local Gaussian SimpleNet

This variant is isolated from the existing `simplenet.py` and
`simplenet_plus.py` implementations.

## Added entry point

`main_gl.py` trains two lightweight heads over one frozen backbone:

- a global branch for a letterboxed whole-image view;
- a local branch for overlapping tiles;
- Gaussian feature negatives during training only;
- median/MAD branch-score calibration followed by pixel-wise max fusion;
- coordinate-aware Hann-weighted stitching for overlapping tiles.

The evaluator reports exact image AUROC and model-only plus end-to-end latency.
MVTec uses Gaussian-smoothed maps and exact pixel AUROC to match the baseline
post-processing. High-resolution datasets use 4096-bin streaming pixel AUROC.
The evaluator intentionally does not compute PRO.

## VisA

The loader accepts the official raw dataset when
`<data-path>/split_csv/1cls.csv` exists. It also accepts an MVTec-style prepared
directory.

```bash
python main_gl.py \
  --dataset visa \
  --data-path /data/VisA \
  --categories candle capsules cashew chewinggum fryum macaroni1 macaroni2 \
               pcb1 pcb2 pcb3 pcb4 pipe_fryum \
  --run-name visa_gl
```

The default global canvas is 512x768. Local source tiles are 512x512 with a
384-pixel stride and are mapped to the backbone's 320x320 input.

## 4ind

Because the private folder schema is not known to the repository, provide a
CSV manifest with these columns:

```text
category,split,label,image,mask
line_a,train,normal,images/train/0001.png,
line_a,test,anomaly,images/test/0002.png,masks/test/0002.png
```

`image` and `mask` may be absolute paths or paths relative to `--data-path`.
Normal test images may leave `mask` empty.

```bash
python main_gl.py \
  --dataset fourind \
  --data-path /data/4ind \
  --manifest-path /data/4ind/manifest.csv \
  --categories line_a \
  --meta-epochs 1 \
  --gan-epochs 1 \
  --run-name fourind_gl
```

The 4ind default global canvas is 128x2048, matching the dataset's 16:1 aspect
ratio. A 16384x1024 image produces 129 local tiles with the default overlap.
Test tiles remain uint8 in host memory and are normalized one inference
mini-batch at a time.

## MVTec AD

MVTec uses the repository's existing loader. The same resized image is passed
to both heads, and its backbone features are computed once per batch.

```bash
python main_gl.py \
  --dataset mvtec \
  --data-path /data/mvtec \
  --categories bottle cable capsule \
  --run-name mvtec_gl
```

Results and independent checkpoints are stored below
`<results-path>/<run-name>/`; reusing the same run name loads the existing
global-local checkpoint.
