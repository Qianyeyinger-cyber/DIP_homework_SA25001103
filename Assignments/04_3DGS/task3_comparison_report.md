# Task 3: Simplified 3DGS vs Official 3DGS

## Setup

- Dataset: `data/chair`
- Resolution: `100 x 100`
  - Simplified implementation: `ColmapDataset(..., downsample_factor=8)`
  - Official implementation: `-r 8`
- Simplified checkpoint: `data/chair/checkpoints/checkpoint_000060.pt`
- Official implementation: `D:\pyProject\GDL_3DGS_STAGE1_ENV\repo\gaussian-splatting`
- Official command:

```bash
python train.py -s D:\pyProject\04_3DGS\data\chair \
  -m D:\pyProject\04_3DGS\data\chair\official_3dgs \
  --iterations 30000 --save_iterations 30000 \
  --test_iterations -1 -r 8 --white_background
```

`chair` images are RGBA synthetic-style PNGs, so the official run uses `--white_background`.

## Results

| Method | Training steps | Time | Speed | Peak VRAM |
|---|---:|---:|---:|---:|
| Simplified PyTorch | benchmark 100 steps | 128.15 s | 0.78 step/s | 7.96 GiB allocated |
| Official 3DGS | 30000 steps | 174.82 s | 171.60 step/s | 358 MiB by `nvidia-smi` |
| Official 3DGS `-r 1` | 30000 steps | 455.36 s | 65.88 step/s | 2166 MiB by `nvidia-smi` |

For the simplified version, 6100 steps are estimated from the measured benchmark as about `7816.87 s` (`~130.3 min`). This matches the observed slow Task 2 training.

| Method | Metric target | L1 lower | PSNR higher | SSIM higher |
|---|---|---:|---:|---:|
| Simplified | black-background full image | 0.0251 | 21.53 | 0.890 |
| Simplified | foreground mask only | 0.1016 | 15.70 | - |
| Official 3DGS | white-composited full image | 0.0683 | 16.98 | 0.658 |
| Official 3DGS | foreground mask only | 0.2326 | 11.35 | - |
| Official 3DGS `-r 1` | white-composited full image | 0.0727 | 16.97 | 0.793 |
| Official 3DGS `-r 1` | foreground mask only | 0.2390 | 11.70 | - |

Point counts:

| Method | Gaussian count |
|---|---:|
| Initial COLMAP points | 13641 |
| Simplified final | 13641 |
| Official final | 28089 |
| Official final `-r 1` | 304358 |

## Discussion

The official implementation is much faster because it uses a CUDA rasterizer with tile-based splatting and visibility-aware rendering. The simplified implementation evaluates all Gaussians over the full image grid in PyTorch, so its cost scales roughly with `N x H x W` and wastes computation on empty pixels.

The official implementation also uses adaptive densification and pruning. It grows from 13641 COLMAP points to 28089 Gaussians, while the simplified version keeps the initial sparse point count fixed. This is the main reason official 3DGS can represent finer geometry and sharper details in normal settings.

In this run, numeric quality is not strictly favorable to the official result because the dataset was produced through COLMAP from RGBA synthetic images. The official COLMAP loader uses the alpha mask during training, but its render output is a normal RGB image on a white background. Therefore black-background metrics penalize the official render heavily. The side-by-side figure is the most useful visual evidence here:

`data/chair/task3_comparison_grid.png`

The simplified renderer matches the black-background target more directly. The official renderer produces clearer object structure in places, but its background/edge compositing differs from the simplified pipeline, so the full-image PSNR is lower under this assignment data path.

## Full Resolution Official Run

The first official run used `-r 8` to match the simplified renderer's `100 x 100` resolution. A second official run was also performed with `-r 1`, i.e. the original `800 x 800` image resolution:

```bash
python train.py -s D:\pyProject\04_3DGS\data\chair \
  -m D:\pyProject\04_3DGS\data\chair\official_3dgs_r1 \
  --iterations 30000 --save_iterations 30000 \
  --test_iterations -1 -r 1 --white_background
```

This run is visibly sharper because the rasterizer directly optimizes and renders at the original image resolution. It also triggers far more densification: the final model contains 304358 Gaussians instead of 28089 in the `-r 8` run. The tradeoff is higher compute and memory: training took 455.36 s, rendering 100 train views took 9.31 s, and peak GPU memory reported by `nvidia-smi` was 2166 MiB.

Full-resolution comparison figure:

`data/chair/task3_comparison_grid_r1.png`

## Key Outputs

- Official model: `data/chair/official_3dgs/point_cloud/iteration_30000/point_cloud.ply`
- Official renders: `data/chair/official_3dgs/train/ours_30000/renders`
- Official `-r 8` train-view video: `data/chair/official_3dgs/render_train_views.mp4`
- Metrics JSON: `data/chair/task3_quality_metrics.json`
- Comparison grid: `data/chair/task3_comparison_grid.png`
- Full-resolution official model: `data/chair/official_3dgs_r1/point_cloud/iteration_30000/point_cloud.ply`
- Full-resolution official renders: `data/chair/official_3dgs_r1/train/ours_30000/renders`
- Full-resolution official train-view video: `data/chair/official_3dgs_r1/render_train_views.mp4`
- Full-resolution comparison grid: `data/chair/task3_comparison_grid_r1.png`
- Official logs: `data/chair/official_3dgs/official_train.log`
