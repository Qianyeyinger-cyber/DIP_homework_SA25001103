# DIP_homework_SA25001103
# Assignment 1 - Image Warping

This repository contains my implementation for **Assignment 1: Image Warping**.
The assignment includes two parts:

1. **Basic Image Geometric Transformation**  
   Implement scale / rotation / translation in `run_global_transform.py`.
2. **Point-Based Image Deformation**  
   Implement point-guided image warping in `run_point_transform.py` using an **RBF-based deformation** method.

---

## Files

```text
Assignments/01_ImageWarping/
├── README.md
├── run_global_transform.py
├── run_point_transform.py
└── assets/
    ├── global_transform_demo.gif
    └── point_warp_demo.gif
```

In my submission, the two core scripts are:

- `run_global_transform.py`  
  Implements composition of **scaling, rotation, horizontal flip, and translation** using affine matrices and `cv2.warpAffine`.

- `run_point_transform.py`  
  Implements **point-guided image deformation** using a **Radial Basis Function (RBF)** interpolation method for sparse control points.

---

## Environment

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Typical dependencies:

- `opencv-python`
- `numpy`
- `gradio`

---

## Part 1: Basic Image Geometric Transformation

This part supports the following global image operations:

- **Scale**
- **Rotation**
- **Translation**
- **Horizontal Flip**

The transformation is implemented by composing affine matrices and applying them to the uploaded image.  
For scale and rotation, the transform is performed **around the image center**.

### Run

```bash
python run_global_transform.py
```

### Demo Result

<p align="center">
  <img src="assets/global_transform_demo.gif" width="600" alt="Global transform demo gif">
</p>

In this demo, the image is continuously transformed with changing scale, rotation, and translation, showing that the global affine transformation pipeline works correctly.

---

## Part 2: Point-Based Image Deformation

This part implements an **RBF-based point-guided warping method**.
The user clicks corresponding **source points** and **target points** interactively:

- Blue points: source points
- Red points: target points
- Green arrows: displacement from source to target

The deformation field is estimated from sparse point correspondences, and then the image is warped using backward mapping.

### Method

I use a **Radial Basis Function (RBF)** interpolation approach:

- Control points are defined by the selected correspondence pairs.
- A smooth displacement field is interpolated from these sparse points.
- The warped image is generated with `cv2.remap` using inverse mapping.

This implementation is suitable for **facial expression editing**, such as dragging the mouth corners upward to synthesize a smile.

### Run

```bash
python run_point_transform.py
```

### Demo Result

<p align="center">
  <img src="assets/point_warp_demo.gif" width="600" alt="Point warp demo gif">
</p>

The demo shows a simple facial example where only a few mouth-region control points are moved, and the whole mouth shape deforms smoothly into a smiling expression.

---

## Implementation Notes

### `run_global_transform.py`

Main ideas:

- Convert image to NumPy array.
- Pad the image to avoid boundary clipping.
- Construct affine transforms for:
  - rotation + scale around image center,
  - optional horizontal flip,
  - translation.
- Compose these transforms into one matrix.
- Apply with `cv2.warpAffine`.

### `run_point_transform.py`

Main ideas:

- Store interactive point correspondences.
- Use sparse source-target pairs as deformation constraints.
- Fit an RBF interpolation to the displacement field.
- Use inverse warping so that each output pixel samples from a valid source location.
- Visualize source points, target points, and arrows directly in the UI.

---

## Summary

This assignment helped me understand two important classes of image warping methods:

1. **Global geometric transformation** based on affine matrices.
2. **Local point-guided deformation** based on smooth interpolation of sparse control points.

The first method is suitable for rigid image motion, while the second method is more flexible for expression editing and local non-rigid deformation.

---

## Acknowledgement

This work is based on the assignment materials and the following references:

- *Image Deformation Using Moving Least Squares*
- *Image Warping by Radial Basis Functions*
- OpenCV Geometric Transformations
- Gradio interactive interface

