#!/usr/bin/env bash
set -euo pipefail

# 0) Sicherstellen, dass wir im Projekt-Root sind
cd "$(dirname "$0")"

# 1) (Optional) venv aktivieren, falls du den Pfad kennst
# source ~/envs/p5/bin/activate

# 2) Submodule ohne Build-Isolation installieren
pip install --no-build-isolation \
    ./submodules/simple-knn \
    ./submodules/fused-ssim \
    ./submodules/diff-gaussian-rasterization

# 3) Restliche Dependencies aus requirements.txt
pip install --upgrade pip
pip install -r requirements.txt