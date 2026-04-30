#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/micromamba"
"$ROOT/bin/micromamba" install -y -n colmap -c conda-forge faiss
