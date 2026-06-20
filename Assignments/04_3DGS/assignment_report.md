# Assignment 4 Report: Simplified 3D Gaussian Splatting

## 1. 实验环境

本次作业在 Windows 本地环境完成，主要使用两个 Python 环境：

| 用途 | 环境 |
|---|---|
| Task 1/2 简化版 3DGS | `D:\Anaconda3\envs\deeplr1\python.exe` |
| Task 3 官方 3DGS | `D:\Anaconda3\envs\gdl_3dgs_stage1_cuda128_20260606_2331\python.exe` |

GPU 为 NVIDIA GeForce RTX 5070 Ti Laptop GPU。Task 1 使用 COLMAP 恢复相机参数与稀疏点云；Task 2 完成作业框架中的 TODO；Task 3 使用已下载的官方 3DGS 实现进行对比实验。

---

## 2. Task 1: Structure-from-Motion with COLMAP

### 2.1 实验目标

Task 1 的目标是从多视角图片中恢复：

- 相机内参 `K`
- 相机外参 `R, t`
- 稀疏 3D 点云

这些输出随后作为 Task 2 中 3D Gaussian 初始化的输入。

作业要求命令为：

```bash
python mvs_with_colmap.py --data_dir data/chair
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

实际执行时，本机 COLMAP 版本不接受脚本中旧参数名 `--SiftExtraction.use_gpu` / `--SiftMatching.use_gpu`，因此使用等价的 COLMAP 命令完成 SfM 流程，并用作业提供的 `debug_mvs_by_projecting_pts.py` 做重投影验证。

### 2.2 执行数据

除了 `chair`，本次也按要求对 `lego` 做了同样的 SfM 与重投影验证。

| 数据集 | 输入图片数 | 注册图片数 | 相机数 | 稀疏 3D 点数 | 重投影验证图 |
|---|---:|---:|---:|---:|---:|
| `chair` | 100 | 100 | 1 | 13641 | 100 |
| `lego` | 100 | 100 | 1 | 5640 | 100 |

主要输出：

```text
data/chair/database.db
data/chair/sparse/0
data/chair/sparse/0_text
data/chair/projections

data/lego/database.db
data/lego/sparse/0
data/lego/sparse/0_text
data/lego/projections
```

### 2.3 结果分析

COLMAP 对两个数据集均成功注册了全部 100 张图片，说明输入视角之间的特征匹配和相机位姿估计是稳定的。`chair` 恢复出 13641 个稀疏点，`lego` 恢复出 5640 个稀疏点。两者点数差异主要来自纹理、几何结构和可匹配特征数量不同。

重投影验证图位于：

- `data/chair/projections`
- `data/lego/projections`

这些图将恢复出的 3D 点投影回各个视角，用来检查相机参数和点云是否与原图对齐。由于 3D 点来自 SfM，它们只覆盖高置信度特征区域，无法直接用于稠密渲染。因此 Task 2 需要将每个稀疏点扩展为 3D Gaussian。

---

## 3. Task 2: Simplified 3D Gaussian Splatting

### 3.1 实现内容

Task 2 需要完成简化版 3DGS 的核心公式，包括：

1. 由旋转和缩放构造 3D 协方差矩阵
2. 将 3D Gaussian 投影到 2D 图像平面
3. 计算 2D Gaussian 在像素处的取值
4. 按深度排序后进行 alpha blending

对应代码修改：

| 文件 | 内容 |
|---|---|
| `gaussian_model.py` | 实现 `Cov = R S S^T R^T` |
| `gaussian_renderer.py` | 实现投影雅可比、2D 协方差、Gaussian 取值和 alpha blending |
| `data_utils.py` | 用纯 Python/PyTorch 替换缺失的 `natsort` 和 `pytorch3d` 依赖 |

### 3.2 关键公式

3D Gaussian 的协方差矩阵按照论文公式 (6) 构造：

```text
Sigma = R S S^T R^T
```

其中 `R` 由单位四元数得到，`S` 是由 3 维 scaling 参数构成的对角矩阵。

投影到图像平面时，按照作业要求使用：

```text
Sigma' = J W Sigma W^T J^T
```

其中：

- `W` 为世界坐标到相机坐标的旋转部分
- `J` 为透视投影对相机坐标的雅可比矩阵

2D Gaussian 在像素 `x` 处的取值为：

```text
f(x) = 1 / (2 pi sqrt(|Sigma'|)) * exp(-0.5 * (x - mu)^T Sigma'^-1 (x - mu))
```

最终渲染使用按深度排序的 alpha blending：

```text
alpha_i = opacity_i * f_i(x)
T_i = product_{j < i}(1 - alpha_j)
C(x) = sum_i T_i alpha_i c_i
```

### 3.3 训练设置

训练命令：

```bash
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints --num_epochs 61
```

训练分辨率来自 `ColmapDataset` 默认参数：

```text
downsample_factor = 8
800 x 800 -> 100 x 100
```

训练输出：

```text
data/chair/checkpoints/checkpoint_000000.pt
data/chair/checkpoints/checkpoint_000020.pt
data/chair/checkpoints/checkpoint_000040.pt
data/chair/checkpoints/checkpoint_000060.pt
data/chair/checkpoints/debug_images
data/chair/checkpoints/debug_rendering.mp4
data/chair/render_mv.mp4
```

其中 `checkpoint_000060.pt` 是后续渲染和对比使用的简化版模型。

### 3.4 训练结果

训练过程中 loss 明显下降：

| 阶段 | 观察到的 loss |
|---|---:|
| 初始 epoch 0 | 约 0.09 |
| epoch 2 | 约 0.048 |
| epoch 15 | 约 0.030 |
| epoch 60 | 约 0.0257 |

最终渲染视频：

- `data/chair/checkpoints/debug_rendering.mp4`
- `data/chair/render_mv.mp4`

### 3.5 质量分析

简化版 3DGS 能够恢复椅子的主要形状、颜色和视角一致性，说明 Task 2 的核心公式和可微渲染流程是有效的。但是结果仍然有明显局限：

1. 没有 adaptive densification  
   简化版始终只使用 COLMAP 初始点数，即 13641 个 Gaussian。稀疏点不足的位置无法自动补点，因此细节和边缘不够完整。

2. 没有 tile-based rasterizer  
   当前实现直接在 PyTorch 中对所有 Gaussian 和所有像素做全量计算，计算复杂度接近 `N x H x W`。这非常慢，也占用大量显存。

3. 表达能力有限  
   简化版使用简单 RGB 颜色，没有官方实现中的 spherical harmonics 视角相关颜色建模，因此高光、遮挡边缘和细节变化表达较弱。

4. 分辨率较低  
   Task 2 默认将图像下采样到 100x100，渲染清晰度自然受限。

简化版最终训练视角指标如下：

| 指标口径 | L1 | PSNR | SSIM |
|---|---:|---:|---:|
| 黑底全图 | 0.0251 | 21.53 | 0.890 |
| 前景 mask | 0.1016 | 15.70 | - |

---

## 4. Task 3: Compare with the Official 3DGS Implementation

### 4.1 实验目标

Task 3 要求使用相同数据集运行官方 3DGS，并从以下三方面对比：

- 渲染质量
- 训练速度
- 显存占用

官方实现位置：

```text
D:\pyProject\GDL_3DGS_STAGE1_ENV\repo\gaussian-splatting
```

### 4.2 官方 3DGS 设置

为了和 Task 2 的 100x100 简化版结果公平比较，先运行了 `-r 8`：

```bash
python train.py -s D:\pyProject\04_3DGS\data\chair \
  -m D:\pyProject\04_3DGS\data\chair\official_3dgs \
  --iterations 30000 --save_iterations 30000 \
  --test_iterations -1 -r 8 --white_background
```

由于原图是 `800 x 800`，`-r 8` 后训练和渲染分辨率为：

```text
100 x 100
```

随后又运行了 `-r 1`，使用原始全分辨率：

```bash
python train.py -s D:\pyProject\04_3DGS\data\chair \
  -m D:\pyProject\04_3DGS\data\chair\official_3dgs_r1 \
  --iterations 30000 --save_iterations 30000 \
  --test_iterations -1 -r 1 --white_background
```

全分辨率为：

```text
800 x 800
```

这里使用 `--white_background`，因为 `chair` 图片是 RGBA synthetic-style PNG。官方 3DGS 对这类数据通常使用白底合成。

### 4.3 速度与显存对比

| 方法 | 分辨率 | 训练步数 | 训练时间 | 速度 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| 简化版 PyTorch | 100x100 | benchmark 100 steps | 128.15 s | 0.78 step/s | 7.96 GiB allocated |
| 官方 3DGS `-r 8` | 100x100 | 30000 | 174.82 s | 171.60 step/s | 358 MiB |
| 官方 3DGS `-r 1` | 800x800 | 30000 | 455.36 s | 65.88 step/s | 2166 MiB |

简化版按照 100 step benchmark 估算，若训练 6100 step 约需要：

```text
7816.87 s ~= 130.3 min
```

官方 `-r 8` 训练 30000 step 只需要约 174.82 s，速度差距非常明显。

### 4.4 Gaussian 数量对比

| 方法 | Gaussian 数量 |
|---|---:|
| COLMAP 初始稀疏点 | 13641 |
| 简化版最终模型 | 13641 |
| 官方 3DGS `-r 8` | 28089 |
| 官方 3DGS `-r 1` | 304358 |

官方 3DGS 的一个核心优势是 adaptive densification。它会根据训练过程中的梯度和可见性信息进行 clone/split/prune，使 Gaussian 数量从初始稀疏点增长到更适合表达表面细节的规模。

全分辨率 `-r 1` 的最终 Gaussian 数量达到 304358，远多于 `-r 8` 的 28089，因此渲染细节明显更丰富。

### 4.5 渲染质量对比

定量结果如下：

| 方法 | 评价口径 | L1 | PSNR | SSIM |
|---|---|---:|---:|---:|
| 简化版 | 黑底全图 | 0.0251 | 21.53 | 0.890 |
| 简化版 | 前景 mask | 0.1016 | 15.70 | - |
| 官方 `-r 8` | 白底合成 GT 全图 | 0.0683 | 16.98 | 0.658 |
| 官方 `-r 8` | 前景 mask | 0.2326 | 11.35 | - |
| 官方 `-r 1` | 白底合成 GT 全图 | 0.0727 | 16.97 | 0.793 |
| 官方 `-r 1` | 前景 mask | 0.2390 | 11.70 | - |

需要注意：这些数值不能简单理解为“官方质量更差”。原因是本作业数据经过 COLMAP 管线读取 RGBA 图片，简化版直接训练黑底 RGB，而官方实现以 white background 方式处理 synthetic-style alpha 图。两者背景和边缘合成口径不同，导致全图 PSNR/SSIM 受到背景处理方式强烈影响。

从视觉结果看：

- `-r 8` 官方结果和简化版分辨率一致，都是 100x100，因此看起来仍然低清晰度。
- `-r 1` 官方结果使用 800x800 原图训练，细节明显更清晰。
- 官方模型在高分辨率下产生大量 Gaussian，椅子纹理、边缘和局部结构更细。
- 简化版在黑底口径下数值更好，但这主要是因为它的训练和评估背景完全一致。

对比图：

- `data/chair/task3_comparison_grid.png`
- `data/chair/task3_comparison_grid_r1.png`

官方训练视角视频：

- `data/chair/official_3dgs/render_train_views.mp4`
- `data/chair/official_3dgs_r1/render_train_views.mp4`

### 4.6 差异来源分析

官方 3DGS 明显快于简化版，核心原因是渲染器不同。

简化版实现中，每个训练 step 都要对所有 Gaussian 和所有像素进行计算。对于 `N` 个 Gaussian、图像大小 `H x W`，计算量接近：

```text
O(N H W)
```

这会在空白区域浪费大量计算。

官方 3DGS 使用 CUDA tile-based rasterizer，只处理实际影响当前 tile 的 Gaussian，并结合深度排序和可见性优化，因此训练和渲染速度远高于纯 PyTorch 实现。

官方质量更强的原因主要包括：

1. Adaptive densification  
   官方会自动增加和裁剪 Gaussian，使模型从稀疏 SfM 点逐步变成更稠密的表面表示。

2. Anisotropic covariance optimization  
   官方优化各向异性 Gaussian，能更好贴合物体表面。

3. Spherical harmonics color  
   官方使用 SH 表达视角相关颜色，比简化版固定 RGB 更强。

4. 高效 rasterization  
   官方 rasterizer 支持高分辨率训练和渲染，`-r 1` 下可以直接输出 800x800 清晰图像。

简化版的价值在于帮助理解 3DGS 的核心数学流程：SfM 初始化、Gaussian 参数化、投影、2D Gaussian 评估和 alpha blending。但它缺少官方系统中决定最终质量和效率的工程模块。

---

## 5. 最终产物列表

Task 1:

```text
data/chair/database.db
data/chair/sparse
data/chair/projections
data/lego/database.db
data/lego/sparse
data/lego/projections
```

Task 2:

```text
data/chair/checkpoints/checkpoint_000060.pt
data/chair/checkpoints/debug_images
data/chair/checkpoints/debug_rendering.mp4
data/chair/render_mv.mp4
```

Task 3:

```text
data/chair/official_3dgs
data/chair/official_3dgs/render_train_views.mp4
data/chair/official_3dgs_r1
data/chair/official_3dgs_r1/render_train_views.mp4
data/chair/task3_quality_metrics.json
data/chair/task3_comparison_grid.png
data/chair/task3_comparison_grid_r1.png
task3_comparison_report.md
```

---

## 6. 总结

本次作业完整完成了从 COLMAP SfM 到简化 3DGS 训练，再到官方 3DGS 对比的流程。

Task 1 中，`chair` 和 `lego` 均成功恢复相机参数和稀疏点云。Task 2 中，简化版 3DGS 能够基于 COLMAP 点云完成基本重建，但受限于无 densification、无 tile rasterizer 和低分辨率训练，清晰度和效率有限。Task 3 中，官方 3DGS 在速度和高分辨率能力上明显更强，尤其是 `-r 1` 训练后能生成更清晰的 800x800 结果，但背景合成方式与简化版不同，评价指标需要按一致背景口径解释。
