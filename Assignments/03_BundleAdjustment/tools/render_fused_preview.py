#!/usr/bin/env python
from pathlib import Path
import argparse

import numpy as np
from PIL import Image


def read_ply(path: Path):
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        n_vertices = next(
            int(line.split()[-1])
            for line in header
            if line.startswith("element vertex")
        )
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        data = np.fromfile(f, dtype=dtype, count=n_vertices)
    pts = np.stack([data["x"], data["y"], data["z"]], axis=1)
    cols = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(
        np.uint8
    )
    return pts, cols


def build_canonical_basis(pts: np.ndarray):
    center = pts.mean(axis=0)
    centered = pts - center

    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Flip the main PCA axis so the head is up in the preview.
    up = -eigvecs[:, 0]

    # Use the third PCA axis as the side-to-side direction.
    right = -eigvecs[:, 2]
    right /= np.linalg.norm(right)

    depth = np.cross(right, up)
    depth /= np.linalg.norm(depth)

    basis = np.stack([right, up, depth], axis=1)
    if np.linalg.det(basis) < 0:
        depth = -depth
        basis = np.stack([right, up, depth], axis=1)
    return center, basis


def render_frame(canon_pts: np.ndarray, colors: np.ndarray, angle: float, size: int, span: float):
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    pts = canon_pts @ rot.T

    # Draw far points first, then near points.
    order = np.argsort(pts[:, 2])
    pts = pts[order]
    colors = colors[order]

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    px = (pts[:, 0] / span + 0.5) * (size - 1)
    py = (0.5 + pts[:, 1] / span) * (size - 1)
    xs = np.clip(np.rint(px).astype(np.int32), 0, size - 1)
    ys = np.clip(np.rint(py).astype(np.int32), 0, size - 1)

    for x, y, color in zip(xs, ys, colors):
        canvas[y, x] = color

    return Image.fromarray(canvas, mode="RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="Input PLY file. Defaults to data/colmap/dense/fused.ply.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Output PNG preview path. Defaults next to the PLY.",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Output GIF preview path. Defaults next to the PLY.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ply_path = args.ply or (root / "data" / "colmap" / "dense" / "fused.ply")
    png_path = args.png or ply_path.with_name("fused_preview.png")
    gif_path = args.gif or ply_path.with_name("fused_preview.gif")

    pts, cols = read_ply(ply_path)

    rng = np.random.default_rng(0)
    sample_size = min(50000, len(pts))
    idx = rng.choice(len(pts), size=sample_size, replace=False)
    pts = pts[idx]
    cols = cols[idx]

    center, basis = build_canonical_basis(pts)
    canon_pts = (pts - center) @ basis

    # Keep the subject fully visible without making the frame too empty.
    width_radius = np.max(np.linalg.norm(canon_pts[:, [0, 2]], axis=1))
    height_radius = np.max(np.abs(canon_pts[:, 1]))
    span = max(width_radius, height_radius) * 1.25

    frames = []
    num_frames = 70
    for i in range(num_frames):
        angle = 2.0 * np.pi * i / num_frames
        frame = render_frame(canon_pts, cols, angle, size=256, span=span)
        frames.append(frame)

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
        optimize=False,
        disposal=2,
    )

    # Save a front-facing still image for the PNG preview.
    png_frame_idx = 20
    png_angle = 2.0 * np.pi * png_frame_idx / num_frames
    png_frame = render_frame(canon_pts, cols, png_angle, size=256, span=span)
    png_frame.save(png_path)
    print(gif_path)
    print(png_path)


if __name__ == "__main__":
    main()
