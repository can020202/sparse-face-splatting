import os
import logging
import shutil
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser("Colmap converter with provided cameras.txt & images.txt")
parser.add_argument("--no_gpu", action='store_true', help="GPU-Nutzung deaktivieren")
parser.add_argument("--skip_matching", action='store_true', help="Feature-Extraction und Matching überspringen")
parser.add_argument("--source_path", "-s", required=True, type=str, help="Pfad zum Projektordner")
parser.add_argument("--camera", default="OPENCV", type=str, help="Kameramodell (z.B. OPENCV)")
parser.add_argument("--colmap_executable", default="", type=str, help="Pfad zur COLMAP-Binärdatei")
parser.add_argument("--resize", action="store_true", help="Bilder resize'n")
parser.add_argument("--magick_executable", default="", type=str, help="Pfad zur ImageMagick-Binärdatei")
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0


import subprocess, sys, logging, textwrap

def run_cmd(cmd):
    print("▶", cmd)
    completed = subprocess.run(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(completed.stdout)
    if completed.returncode != 0:
        print("=== STDERR ===", file=sys.stderr)
        print(completed.stderr, file=sys.stderr)
        sys.exit(completed.returncode)




database_path = os.path.join(args.source_path, "distorted", "database.db")
input_image_path = os.path.join(args.source_path, "input")
model_txt_path = os.path.join(args.source_path, "model_txt")
triangulated_model_path = os.path.join(args.source_path, "triangulated_model")
sparse_dir = os.path.join(args.source_path, "sparse")
images_dir = os.path.join(args.source_path, "images")

if not args.skip_matching:
    os.makedirs(os.path.join(args.source_path, "distorted", "sparse"), exist_ok=True)

    # Feature Extraction
    feat_extraction_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {database_path} "
        f"--image_path {input_image_path} "
        f"--ImageReader.camera_model PINHOLE "
        f"--ImageReader.single_camera 1 "
        f"--SiftExtraction.use_gpu {use_gpu} "
        f"--SiftExtraction.max_image_size 2000 "
        f"--SiftExtraction.max_num_features 10000 "
        f"--SiftExtraction.peak_threshold 0.005 "
        f"--SiftExtraction.edge_threshold 5"
    )

    run_cmd(feat_extraction_cmd)

    # Feature Matching
    feat_matching_cmd = (
        f"{colmap_command} exhaustive_matcher "
        f"--database_path {database_path} "
        f"--SiftMatching.use_gpu {use_gpu} "
        f"--SiftMatching.guided_matching 1 "
        f"--SiftMatching.max_num_matches 100000 "
        f"--SiftMatching.max_ratio 0.99"
    )
    run_cmd(feat_matching_cmd)

os.makedirs(model_txt_path, exist_ok=True)
points3D_file = os.path.join(model_txt_path, "points3D.txt")
if not os.path.exists(points3D_file):
    with open(points3D_file, "w") as f:
        f.write("")


os.makedirs(triangulated_model_path, exist_ok=True)

point_triangulator_cmd = (
    f"{colmap_command} point_triangulator "
    f"--database_path {database_path} "
    f"--image_path {input_image_path} "
    f"--input_path {model_txt_path} "
    f"--output_path {triangulated_model_path}"
)
run_cmd(point_triangulator_cmd)

model0_dir = os.path.join(triangulated_model_path, "0")
os.makedirs(model0_dir, exist_ok=True)
for fname in ("cameras.bin", "images.bin", "points3D.bin"):
    src = os.path.join(triangulated_model_path, fname)
    dst = os.path.join(model0_dir, fname)
    if os.path.exists(src):
        shutil.move(src, dst)

img_undist_cmd = (
    f"{colmap_command} image_undistorter "
    f"--image_path {input_image_path} "
    f"--input_path {os.path.join(triangulated_model_path, '0')} "
    f"--output_path {args.source_path} "
    f"--output_type COLMAP"
)
run_cmd(img_undist_cmd)

if os.path.exists(sparse_dir):
    files = os.listdir(sparse_dir)
    os.makedirs(os.path.join(sparse_dir, "0"), exist_ok=True)
    for file in files:
        if file == "0":
            continue
        src_file = os.path.join(sparse_dir, file)
        dst_file = os.path.join(sparse_dir, "0", file)
        shutil.move(src_file, dst_file)

if args.resize:
    print("Copying and resizing images...")
    images_2_dir = os.path.join(args.source_path, "images_2")
    images_4_dir = os.path.join(args.source_path, "images_4")
    images_8_dir = os.path.join(args.source_path, "images_8")
    os.makedirs(images_2_dir, exist_ok=True)
    os.makedirs(images_4_dir, exist_ok=True)
    os.makedirs(images_8_dir, exist_ok=True)

    for file in os.listdir(images_dir):
        source_file = os.path.join(images_dir, file)
        dest_file_2 = os.path.join(images_2_dir, file)
        shutil.copy2(source_file, dest_file_2)
        run_cmd(f"{magick_command} mogrify -resize 50% {dest_file_2}")

        dest_file_4 = os.path.join(images_4_dir, file)
        shutil.copy2(source_file, dest_file_4)
        run_cmd(f"{magick_command} mogrify -resize 25% {dest_file_4}")

        dest_file_8 = os.path.join(images_8_dir, file)
        shutil.copy2(source_file, dest_file_8)
        run_cmd(f"{magick_command} mogrify -resize 12.5% {dest_file_8}")

print("Done.")
