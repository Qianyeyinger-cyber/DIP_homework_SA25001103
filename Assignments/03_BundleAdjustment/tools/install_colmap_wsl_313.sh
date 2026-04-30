#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/micromamba"
"$ROOT/bin/micromamba" create -y -r "$ROOT" -n colmap313 -c conda-forge python=3.13 colmap faiss
