# Partial Cross-Entropy Segmentation Notebook

This repository includes a research-style notebook, `partial_ce_segment.ipynb`, that demonstrates **point-supervised semantic segmentation** for synthetic remote sensing imagery using a U-Net and Partial Focal Cross-Entropy loss.

## Notebook Goal

Train a segmentation model when only sparse point annotations are available (instead of full masks), and study how performance changes with:

- Point label density
- Focal loss parameter $\gamma$
- Network depth (shallow vs deep U-Net)
- Semi-supervised pseudo-labeling

## What the Notebook Contains

1. **Environment setup** and imports (PyTorch, torchvision, sklearn, matplotlib, seaborn)
2. **Synthetic remote sensing data generation**
3. **Point-label simulation** from dense ground-truth masks
4. **Partial Focal CE loss** implementation (uses only labeled pixels)
5. **Configurable U-Net** implementation
6. **Training & evaluation pipeline**
7. **Four controlled experiments**
8. **Plots, qualitative predictions, and final summary report**

## Experiments

### Experiment 1 — Point Label Density
Compares density values: `1%`, `5%`, `10%`, `20%`.

### Experiment 2 — Focal Loss Gamma
Compares gamma values: `0`, `1`, `2`, `5`.

### Experiment 3 — Model Depth
Compares U-Net depth: `2` vs `3`.

### Experiment 4 — Semi-Supervised Learning
Compares:
- Supervised-only training
- Semi-supervised training with pseudo-labels (confidence threshold `0.85`)

## Key Method: Partial Focal Cross-Entropy

Only labeled points contribute to the loss:

$$
\mathcal{L}_{pfCE} = \frac{\sum_i \mathcal{L}_{focal}(\hat{y}_i, y_i)\cdot\mathbf{1}[p_i\ge0]}{\sum_i \mathbf{1}[p_i\ge0]}
$$

Where unlabeled pixels are marked with `-1` and ignored.

## Requirements

Install dependencies used in the notebook:

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm Pillow seaborn
```

> The first notebook cell already installs these packages with `pip`.

## How to Run

1. Open `partial_ce_segment.ipynb`.
2. Select a Python kernel (preferably with GPU support if available).
3. Run cells from top to bottom.
4. Expect training to take time (multiple experiments × multiple epochs).

## Main Configuration Defaults

- `NUM_CLASSES = 6`
- `IMG_SIZE = 64`
- `BATCH_SIZE = 16`
- `EPOCHS = 25`
- `LR = 1e-3`
- `NUM_TRAIN = 400`
- `NUM_VAL = 100`

You can adjust these values in the configuration cell to trade off speed vs quality.

## Outputs Produced

The notebook saves figures such as:

- `sample_visualisation.png`
- `exp1_point_density.png`, `exp1_bar.png`
- `exp2_gamma.png`, `exp2_bar.png`
- `exp3_depth.png`, `exp3_bar.png`
- `exp4_semi.png`, `exp4_bar.png`
- `predictions.png`
- `overall_summary.png`

## Notes

- Dataset is **synthetic**, designed for controlled comparison of methods.
- Full segmentation masks are used for evaluation, but training loss only uses sparse point labels.
- Results are reproducible via fixed random seed (`SEED = 42`).

---

If you want, I can also generate a shorter one-page `README` version optimized for GitHub project previews.