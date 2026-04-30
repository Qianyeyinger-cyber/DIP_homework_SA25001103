#!/usr/bin/env python
"""
Task 1: 使用 PyTorch 实现 Bundle Adjustment。

输入:
- data/points2d.npz
- data/points3d_colors.npy

输出:
- outputs/task1/loss_curve.png
- outputs/task1/reconstruction.obj
- outputs/task1/result.npz
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Bundle Adjustment")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/task1", help="结果输出目录"
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    parser.add_argument("--iters", type=int, default=300, help="优化迭代次数")
    parser.add_argument("--lr", type=float, default=5e-3, help="基础学习率")
    parser.add_argument(
        "--fov-deg", type=float, default=60.0, help="焦距初始化用的视场角"
    )
    parser.add_argument(
        "--depth", type=float, default=2.5, help="相机初始到物体的距离"
    )
    parser.add_argument(
        "--init-point-scale", type=float, default=0.08, help="3D 点初始化尺度"
    )
    parser.add_argument(
        "--init-yaw-deg",
        type=float,
        default=65.0,
        help="相机初始绕 Y 轴摆开的角度范围",
    )
    parser.add_argument(
        "--log-every", type=int, default=20, help="多少次迭代打印一次日志"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_data(data_dir):
    data_dir = Path(data_dir)
    points2d = np.load(str(data_dir / "points2d.npz"))
    colors = np.load(str(data_dir / "points3d_colors.npy")).astype(np.float32)
    keys = sorted(points2d.files)

    pt_indices = []
    obs_uv = []
    visible_counts = []
    for key in keys:
        arr = points2d[key].astype(np.float32)
        vis = arr[:, 2] > 0.5
        idx = np.flatnonzero(vis).astype(np.int64)
        uv = arr[vis, :2].astype(np.float32)
        pt_indices.append(idx)
        obs_uv.append(uv)
        visible_counts.append(int(idx.shape[0]))
    return keys, pt_indices, obs_uv, visible_counts, colors


def euler_xyz_to_matrix(euler):
    """
    把 (N, 3) 的 Euler 角转成旋转矩阵。
    这里按 X -> Y -> Z 的顺序组合。
    """

    x = euler[:, 0]
    y = euler[:, 1]
    z = euler[:, 2]

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    ones = torch.ones_like(cx)
    zeros = torch.zeros_like(cx)

    rx = torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cx, -sx), dim=-1),
            torch.stack((zeros, sx, cx), dim=-1),
        ),
        dim=-2,
    )
    ry = torch.stack(
        (
            torch.stack((cy, zeros, sy), dim=-1),
            torch.stack((zeros, ones, zeros), dim=-1),
            torch.stack((-sy, zeros, cy), dim=-1),
        ),
        dim=-2,
    )
    rz = torch.stack(
        (
            torch.stack((cz, -sz, zeros), dim=-1),
            torch.stack((sz, cz, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )

    return torch.bmm(rz, torch.bmm(ry, rx))


class BundleAdjustmentModel(nn.Module):
    def __init__(self, n_views, n_points, initial_f, initial_depth, init_point_scale, init_yaw_deg):
        super(BundleAdjustmentModel, self).__init__()
        self.log_f = nn.Parameter(torch.tensor(math.log(initial_f), dtype=torch.float32))

        euler = torch.zeros(n_views, 3, dtype=torch.float32)
        if n_views > 1:
            yaw = torch.linspace(
                math.radians(init_yaw_deg),
                math.radians(-init_yaw_deg),
                n_views,
            )
            euler[:, 1] = yaw
        self.euler = nn.Parameter(euler)

        translation = torch.zeros(n_views, 3, dtype=torch.float32)
        translation[:, 2] = -float(initial_depth)
        self.translation = nn.Parameter(translation)

        points = init_point_scale * torch.randn(n_points, 3, dtype=torch.float32)
        self.points = nn.Parameter(points)

    def focal(self):
        return torch.exp(self.log_f)


def project_points(points, rotation, translation, focal, cx, cy):
    camera_points = torch.matmul(points, rotation.t()) + translation
    u = -focal * camera_points[:, 0] / camera_points[:, 2] + cx
    v = focal * camera_points[:, 1] / camera_points[:, 2] + cy
    return torch.stack((u, v), dim=-1)


def save_obj(path, points, colors):
    points = points.detach().cpu().numpy()
    colors = np.asarray(colors, dtype=np.float32)
    colors = np.clip(colors, 0.0, 1.0)

    with path.open("w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            f.write(
                "v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    float(c[0]),
                    float(c[1]),
                    float(c[2]),
                )
            )


def save_loss_curve(path, losses):
    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Reprojection Error")
    plt.title("Bundle Adjustment Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=160)
    plt.close()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    keys, pt_indices_np, obs_uv_np, visible_counts, colors = load_data(data_dir)
    n_views = len(keys)
    n_points = colors.shape[0]
    image_h = 1024
    image_w = 1024
    cx = torch.tensor(image_w / 2.0, dtype=torch.float32, device=device)
    cy = torch.tensor(image_h / 2.0, dtype=torch.float32, device=device)
    initial_f = image_h / (2.0 * math.tan(math.radians(args.fov_deg) / 2.0))

    pt_indices = []
    obs_uv = []
    for idx_np, uv_np in zip(pt_indices_np, obs_uv_np):
        pt_indices.append(torch.from_numpy(idx_np).to(device))
        obs_uv.append(torch.from_numpy(uv_np).to(device))

    total_observations = float(sum(visible_counts))
    total_observations_tensor = torch.tensor(total_observations, dtype=torch.float32, device=device)

    model = BundleAdjustmentModel(
        n_views=n_views,
        n_points=n_points,
        initial_f=initial_f,
        initial_depth=args.depth,
        init_point_scale=args.init_point_scale,
        init_yaw_deg=args.init_yaw_deg,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": [model.log_f], "lr": args.lr * 0.1},
            {"params": [model.euler, model.translation], "lr": args.lr},
            {"params": [model.points], "lr": args.lr * 0.2},
        ]
    )

    losses = []
    print("开始优化 Bundle Adjustment ...")
    print("可见观测数:", int(total_observations))
    print("初始焦距:", float(model.focal().detach().cpu().item()))

    for step in range(1, args.iters + 1):
        optimizer.zero_grad(set_to_none=True)

        focal = model.focal()
        rotations = euler_xyz_to_matrix(model.euler)

        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        for view_idx in range(n_views):
            idx = pt_indices[view_idx]
            uv = obs_uv[view_idx]
            pts = model.points.index_select(0, idx)
            proj = project_points(
                pts,
                rotations[view_idx],
                model.translation[view_idx],
                focal,
                cx,
                cy,
            )
            diff = proj - uv
            loss_sum = loss_sum + torch.sum(diff * diff)

        loss = loss_sum / total_observations_tensor
        if not torch.isfinite(loss):
            raise RuntimeError("loss 变成了非有限值，请检查初始化或学习率")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)

        if step == 1 or step % args.log_every == 0 or step == args.iters:
            print(
                "iter {:4d}/{:4d} | loss {:.6f} | f {:.3f}".format(
                    step, args.iters, loss_value, float(model.focal().detach().cpu().item())
                )
            )

    final_points = model.points.detach().cpu()
    final_f = float(model.focal().detach().cpu().item())
    final_euler = model.euler.detach().cpu()
    final_translation = model.translation.detach().cpu()

    obj_path = output_dir / "reconstruction.obj"
    loss_path = output_dir / "loss_curve.png"
    result_path = output_dir / "result.npz"

    save_obj(obj_path, final_points, colors)
    save_loss_curve(loss_path, losses)
    np.savez(
        str(result_path),
        focal=final_f,
        euler=final_euler.numpy(),
        translation=final_translation.numpy(),
        points=final_points.numpy(),
        losses=np.asarray(losses, dtype=np.float32),
    )

    print("完成。")
    print("输出:")
    print(" -", str(obj_path))
    print(" -", str(loss_path))
    print(" -", str(result_path))


if __name__ == "__main__":
    main()
