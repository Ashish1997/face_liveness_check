# face_liveness_check

## Lightweight face landmark detection (OpenCV-only)

This repo includes a lightweight face landmark detector using OpenCV's LBF
facemark model and Haar cascades, designed to run on edge devices without
MediaPipe.

### Setup

Install dependencies (OpenCV contrib needed for facemark):

```
python3 -m pip install -r requirements.txt
```

### Run on images in current directory

Default is to read images from `images/input` and write outputs to
`images/output`. You must provide the LBF model file once.

```
python3 landmark_detect.py --draw --lbf-model models/lbfmodel.yaml
```

Download the LBF model (once) and place it at `models/lbfmodel.yaml`:
- https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml

### Options

- `--input-dir` Directory containing images (default: `images/input`)
- `--output-dir` Output directory (default: `images/output`)
- `--max-faces` Maximum faces per image (default: 1)
- `--face-cascade` Haar cascade path or `opencv_default`
- `--lbf-model` Path to LBF model (default: `models/lbfmodel.yaml`)
- `--min-detection-confidence` Detection confidence (default: 0.5)
- `--min-tracking-confidence` Tracking confidence (default: 0.5)
- `--draw` Save annotated images with landmarks