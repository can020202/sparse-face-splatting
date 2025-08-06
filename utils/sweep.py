
import argparse
import os
import sys
import subprocess
import runpy
from pathlib import Path
from typing import List

import wandb

input_path_params_dir = "cam_params_big_brett"
val_pic_list         = ["val.jpg"]

PROJECT      = "gaussian-splatting"
TRAIN_SCRIPT = "train.py"
EVAL_INTERVAL         = 50
DENSIFY_FROM_ITER     = 50
DENSIFY_UNTIL_ITER    = 25_000
TORCHDYNAMO_DISABLE   = "1"

SWEEP_ID_FILE        = Path(".sweep_id")

def test_iterations_list(iterations: int, eval_interval: int = EVAL_INTERVAL) -> List[int]:
    lst = list(range(eval_interval, iterations + 1, eval_interval))
    if lst[-1] != iterations:
        lst.append(iterations)
    return lst


def prepare_once(name_dir_data: str):
    DATA_DIR             = Path("data")
    SOURCE_PARAMS_DIR    = DATA_DIR / input_path_params_dir
    INPUT_PATH_PARAMS    = SOURCE_PARAMS_DIR / "distorted" / "sparse" / "0"

    SOURCE_WP            = DATA_DIR / name_dir_data
    VAL_PATH             = SOURCE_WP / "sparse" / "0"
    OUTPUT_TXT_DIR       = SOURCE_WP / "model_txt"
    MODEL_PATH_WP        = Path("outputs") / f"{name_dir_data}_model"
    IMAGES_TXT           = OUTPUT_TXT_DIR / "images.txt"
    PREP_DONE            = OUTPUT_TXT_DIR / ".prep_done"

    if PREP_DONE.exists():
        print("✔ Datenvorbereitung bereits erledigt.")
        return

    print("▶ Starte Datenvorbereitung…")
    OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        sys.executable, "utils/make_val_file.py", str(VAL_PATH), *val_pic_list
    ], check=True)

    subprocess.run([
        sys.executable, "utils/sfm.py", "--source_path", str(SOURCE_PARAMS_DIR)
    ], check=True)

    subprocess.run([
        sys.executable, "utils/bin_to_txt.py", "-i", str(INPUT_PATH_PARAMS),
        "-o", str(OUTPUT_TXT_DIR)
    ], check=True)

    subprocess.run([
        sys.executable, "utils/update_images_txt.py",
        "--original", str(IMAGES_TXT), "--output", str(IMAGES_TXT)
    ], check=True)

    subprocess.run([
        sys.executable, "utils/sfm_with_params.py", "--source_path", str(SOURCE_WP)
    ], check=True)

    subprocess.run([
        sys.executable, "utils/make_depth_scale.py", "--base_dir", str(SOURCE_WP),
        "--depths_dir", str(SOURCE_WP / "depth")
    ], check=True)

    PREP_DONE.touch()
    print("✔ Datenvorbereitung abgeschlossen.")


SWEEP_CFG = dict(
    method="bayes",
    metric=dict(name="test_psnr", goal="maximize"),
    parameters=dict(
        sh_degree=dict(values=[2, 3, 4, 5]),
        iterations=dict(values=[2250, 3000, 4000, 5000]),
        percent_dense=dict(distribution="log_uniform_values", min=1e-4, max=2e-2),
        densification_interval=dict(values=[25, 50, 100]),
        densify_grad_threshold=dict(distribution="log_uniform_values", min=1e-4, max=5e-2),
        lambda_dssim=dict(distribution="log_uniform_values", min=0.1, max=1.0),

        antialiasing=dict(values=[True, False]),
        depth_l1_weight_init=dict(distribution="log_uniform_values", min=0.001, max=1.0),
        depth_l1_weight_final=dict(distribution="log_uniform_values", min=0.1, max=20.0),

        densify_from_iter=dict(values=[50, 100, 150]),
        densify_until_ratio=dict(values=[1.0, 0.95, 0.9, 0.8]),
    ),
)

def one_run(name_dir_data: str):
    os.environ["TORCHDYNAMO_DISABLE"] = TORCHDYNAMO_DISABLE
    prepare_once(name_dir_data)

    DATA_DIR             = Path("data")
    SOURCE_WP            = DATA_DIR / name_dir_data
    MODEL_PATH_WP        = Path("outputs") / f"{name_dir_data}_model"

    run = wandb.init(project=PROJECT, config=SWEEP_CFG["parameters"])
    cfg = run.config

    test_iters = test_iterations_list(cfg.iterations)

    densify_until_iter = int(cfg.iterations * cfg.densify_until_ratio)

    sys.argv = [
        TRAIN_SCRIPT,
        "--source_path", str(SOURCE_WP),
        "--model_path", str(MODEL_PATH_WP),
        "--iterations", str(cfg.iterations),
        "--sh_degree", str(cfg.sh_degree),
        "--percent_dense", str(cfg.percent_dense),
        "--densification_interval", str(cfg.densification_interval),
        "--densify_grad_threshold", str(cfg.densify_grad_threshold),
        "--lambda_dssim", str(cfg.lambda_dssim),

        "--densify_from_iter", str(cfg.densify_from_iter),
        "--densify_until_iter", str(densify_until_iter),

        "--disable_viewer",
        "--eval",
        "--test_iterations", *map(str, test_iters),
        "--depths", "depth",
        "--mask_folder", "segmantation",
        "--white_background",

        "--antialiasing" if cfg.antialiasing else "",
        "--depth_l1_weight_init", str(cfg.depth_l1_weight_init),
        "--depth_l1_weight_final", str(cfg.depth_l1_weight_final),
    ]

    sys.argv = [arg for arg in sys.argv if arg != ""]

    runpy.run_path(TRAIN_SCRIPT, run_name="__main__")
    run.finish()

def main():
    parser = argparse.ArgumentParser(description="Starte einen W&B-Sweep für Gaussian Splatting.")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Wie viele Runs dieser Agent abarbeiten soll (0 = unendlich)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bestehenden Sweep aus .sweep_id fortsetzen"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
        help="Name des Datenordners (z. B. 'moritz_without_brett')"
    )

    args = parser.parse_args()

    if args.data_name is None:
        print("⚠ Kein --data_name angegeben! Verwende Standard 'moritz_without_brett'.")
        name_dir_data = "moritz_without_brett"
    else:
        name_dir_data = args.data_name

    if args.resume and SWEEP_ID_FILE.exists():
        sweep_id = SWEEP_ID_FILE.read_text().strip()
        print(f"▶ Resume Sweep {sweep_id}")
    else:
        sweep_id = wandb.sweep(SWEEP_CFG, project=PROJECT)
        SWEEP_ID_FILE.write_text(sweep_id)
        print(f"▶ Created Sweep {sweep_id}")

    count_arg = None if args.count == 0 else args.count
    wandb.agent(sweep_id, function=lambda: one_run(name_dir_data), count=count_arg)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
