@echo off
REM Stelle sicher, dass deine venv aktiviert ist, wenn du eine nutzt
REM    (z.B. .venv\Scripts\activate.bat)

REM 2) Submodule ohne Build-Isolation installieren
pip install --no-build-isolation git+https://github.com/graphdeco-inria/gaussian-splatting.git@main#subdirectory=submodules/simple-knn
pip install --no-build-isolation git+https://github.com/graphdeco-inria/gaussian-splatting.git@main#subdirectory=submodules/fused-ssim
pip install --no-build-isolation git+https://github.com/graphdeco-inria/gaussian-splatting.git@main#subdirectory=submodules/diff-gaussian-rasterization

REM 3) Restliche Dependencies
pip install -r requirements.txt
