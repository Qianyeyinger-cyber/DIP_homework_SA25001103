#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/micromamba"
ARCHIVE="${ARCHIVE:-$PWD/tools/micromamba-linux-64.tar.bz2}"
mkdir -p "$ROOT"

if [ ! -x "$ROOT/bin/micromamba" ]; then
    if [ ! -f "$ARCHIVE" ]; then
        echo "Archive not found: $ARCHIVE"
        exit 1
    fi
    tar -xjf "$ARCHIVE" -C "$ROOT" bin/micromamba
fi

"$ROOT/bin/micromamba" --version
"$ROOT/bin/micromamba" create -y -r "$ROOT" -n colmap -c conda-forge colmap
