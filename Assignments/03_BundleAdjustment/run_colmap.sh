#!/bin/bash
# COLMAP 3D reconstruction pipeline
# Usage: bash run_colmap.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/data}"
IMAGE_PATH="$DATASET_PATH/images"
COLMAP_PATH="$DATASET_PATH/colmap"
SPARSE_ONLY="${SPARSE_ONLY:-0}"

if [ -n "${COLMAP_BIN:-}" ]; then
    :
elif command -v colmap >/dev/null 2>&1; then
    COLMAP_BIN="colmap"
elif [ -x "$SCRIPT_DIR/tools/colmap_micromamba.sh" ]; then
    COLMAP_BIN="$SCRIPT_DIR/tools/colmap_micromamba.sh"
elif [ -x "$SCRIPT_DIR/tools/colmap-4.0.4-nocuda/bin/colmap.exe" ]; then
    COLMAP_BIN="$SCRIPT_DIR/tools/colmap-4.0.4-nocuda/bin/colmap.exe"
elif [ -x "$SCRIPT_DIR/tools/colmap-4.0.4-nocuda/bin/colmap" ]; then
    COLMAP_BIN="$SCRIPT_DIR/tools/colmap-4.0.4-nocuda/bin/colmap"
else
    echo "COLMAP executable not found."
    echo "Set COLMAP_BIN or place COLMAP under tools/colmap-4.0.4-nocuda/bin/."
    exit 1
fi

USE_WINDOWS_PATHS=0
if [[ "$COLMAP_BIN" == *.exe || "$COLMAP_BIN" == *.bat ]]; then
    USE_WINDOWS_PATHS=1
fi

if [ "$USE_WINDOWS_PATHS" = "1" ]; then
    IMAGE_PATH_ARG="$(wslpath -w "$IMAGE_PATH")"
    COLMAP_PATH_ARG="$(wslpath -w "$COLMAP_PATH")"
    PATH_SEP="\\"
else
    IMAGE_PATH_ARG="$IMAGE_PATH"
    COLMAP_PATH_ARG="$COLMAP_PATH"
    PATH_SEP="/"
fi

mkdir -p "$COLMAP_PATH/sparse"
mkdir -p "$COLMAP_PATH/dense"

echo "=== Step 1: Feature Extraction ==="
"$COLMAP_BIN" feature_extractor \
    --database_path "$COLMAP_PATH_ARG${PATH_SEP}database.db" \
    --image_path "$IMAGE_PATH_ARG" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

echo "=== Step 2: Feature Matching ==="
"$COLMAP_BIN" exhaustive_matcher \
    --database_path "$COLMAP_PATH_ARG${PATH_SEP}database.db"

echo "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
"$COLMAP_BIN" mapper \
    --database_path "$COLMAP_PATH_ARG${PATH_SEP}database.db" \
    --image_path "$IMAGE_PATH_ARG" \
    --output_path "$COLMAP_PATH_ARG${PATH_SEP}sparse"

if [ "$SPARSE_ONLY" = "1" ]; then
    echo "=== Sparse only mode enabled, stop here ==="
    echo "Sparse: $COLMAP_PATH/sparse/0/"
    exit 0
fi

echo "=== Step 4: Image Undistortion ==="
"$COLMAP_BIN" image_undistorter \
    --image_path "$IMAGE_PATH_ARG" \
    --input_path "$COLMAP_PATH_ARG${PATH_SEP}sparse${PATH_SEP}0" \
    --output_path "$COLMAP_PATH_ARG${PATH_SEP}dense"

echo "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
"$COLMAP_BIN" patch_match_stereo \
    --workspace_path "$COLMAP_PATH_ARG${PATH_SEP}dense" \
    --PatchMatchStereo.max_image_size 2000 \
    --PatchMatchStereo.geom_consistency false

echo "=== Step 6: Stereo Fusion ==="
"$COLMAP_BIN" stereo_fusion \
    --workspace_path "$COLMAP_PATH_ARG${PATH_SEP}dense" \
    --input_type photometric \
    --StereoFusion.min_num_pixels 3 \
    --output_path "$COLMAP_PATH_ARG${PATH_SEP}dense${PATH_SEP}fused.ply"


echo "=== Done! ==="
echo "Results:"
echo "  Sparse: $COLMAP_PATH/sparse/0/"
echo "  Dense:  $COLMAP_PATH/dense/fused.ply"
