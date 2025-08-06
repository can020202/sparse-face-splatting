#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import torch

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_sapiens = os.path.join(script_dir, "Sapiens-Pytorch-Inference")

    parser = argparse.ArgumentParser(
        description="Erstellt Depth- und Segmentation-Maps für alle Bilder in einem Ordner.\n"
                    "Die Ausgabedateien werden immer als PNG gespeichert."
    )
    parser.add_argument(
        "input_dir",
        help="Pfad zum Ordner mit den Eingabebildern (z. B. JPG/PNG)."
    )
    parser.add_argument(
        "output_depth_dir",
        help="Pfad zum Ordner für die ausgegebenen Depth-Maps."
    )
    parser.add_argument(
        "output_seg_dir",
        help="Pfad zum Ordner für die ausgegebenen Segmentierungs-Maps."
    )
    parser.add_argument(
        "--sapiens_dir",
        default=default_sapiens,
        help=f"Pfad zum Sapiens-Pytorch-Inference-Ordner (Standard: '{default_sapiens}'). "
             "Die Modelle werden automatisch in '…/Sapiens-Pytorch-Inference/models/' gespeichert."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Gerät, auf dem die Modelle laufen: 'gpu' (CUDA) oder 'cpu' (Standard: gpu)."
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    out_depth_dir = os.path.abspath(args.output_depth_dir)
    out_seg_dir = os.path.abspath(args.output_seg_dir)
    sapiens_dir = os.path.abspath(args.sapiens_dir)
    device_choice = args.device

    if not os.path.isdir(sapiens_dir):
        raise FileNotFoundError(f"Der Ordner '{sapiens_dir}' wurde nicht gefunden.")
    if not os.path.isdir(os.path.join(sapiens_dir, "sapiens_inference")):
        raise FileNotFoundError(
            f"Im angegebenen Sapiens-Ordner fehlt das Verzeichnis 'sapiens_inference'.\n"
            f"Bitte stelle sicher, dass '{sapiens_dir}/sapiens_inference' existiert."
        )

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input-Ordner nicht gefunden: {input_dir}")
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_seg_dir, exist_ok=True)

    sys.path.insert(0, sapiens_dir)

    from sapiens_inference import (
        SapiensDepth,
        SapiensDepthType,
        SapiensSegmentation,
        SapiensSegmentationType,
        SapiensConfig
    )

    config = SapiensConfig()
    config.depth_type = SapiensDepthType.DEPTH_1B
    config.segmentation_type = SapiensSegmentationType.SEGMENTATION_1B

    if device_choice == "gpu":
        if torch.cuda.is_available():
            config.device = "cuda"
        else:
            print("Warnung: Keine GPU mit CUDA gefunden. Verwende stattdessen CPU.")
            config.device = "cpu"
    else:
        config.device = "cpu"

    orig_cwd = os.getcwd()
    try:
        os.makedirs(os.path.join(sapiens_dir, "models"), exist_ok=True)
        os.chdir(sapiens_dir)
        depth_predictor = SapiensDepth(config.depth_type, config.device, config.dtype)
        seg_predictor = SapiensSegmentation(config.segmentation_type, config.device, config.dtype)
    finally:
        os.chdir(orig_cwd)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for fname in os.listdir(input_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in extensions:
            continue

        img_path = os.path.join(input_dir, fname)
        pil = Image.open(img_path).convert("RGB")
        rgb_np = np.array(pil)
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        H, W = bgr_np.shape[:2]

        seg_logits = seg_predictor(bgr_np)
        if isinstance(seg_logits, torch.Tensor):
            seg_map = seg_logits.squeeze().cpu().numpy()
        else:
            seg_map = seg_logits

        if seg_map.shape != (H, W):
            seg_map = cv2.resize(seg_map, (W, H), interpolation=cv2.INTER_LINEAR)
        human_mask = (seg_map > 0.5).astype(np.uint8)

        depth_raw = depth_predictor(bgr_np)
        if isinstance(depth_raw, torch.Tensor):
            depth_np = depth_raw.squeeze().cpu().numpy()
        else:
            depth_np = depth_raw

        d_min, d_max = np.nanmin(depth_np), np.nanmax(depth_np)
        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)

        modified_depth = np.where(human_mask == 1, depth_norm, 1.0)

        modified_8u = (modified_depth * 255).astype(np.uint8)
        depth_color_bgr = cv2.applyColorMap(modified_8u, cv2.COLORMAP_TURBO)

        seg_mask_8u = (human_mask * 255).astype(np.uint8)
        seg_out_path = os.path.join(out_seg_dir, f"{name}.png")
        cv2.imwrite(seg_out_path, seg_mask_8u)

        depth_out_path = os.path.join(out_depth_dir, f"{name}.png")
        cv2.imwrite(depth_out_path, depth_color_bgr)

        print(f"Processed: {fname} → Seg: {seg_out_path}, Depth: {depth_out_path}")

if __name__ == "__main__":
    main()
