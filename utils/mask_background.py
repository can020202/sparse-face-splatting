import os
import argparse
import numpy as np
from PIL import Image


def process_folders(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mask_map = {}
    for m in os.listdir(mask_dir):
        name, ext = os.path.splitext(m)
        if ext.lower() in (".png", ".jpg", ".jpeg"):
            mask_map[name] = m

    for img_filename in os.listdir(image_dir):
        base, img_ext = os.path.splitext(img_filename)
        if img_ext.lower() not in (".png", ".jpg", ".jpeg"):

            continue

        if base not in mask_map:
            print(f"Keine Maske gefunden für Bild: {img_filename}")
            continue

        image_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_map[base])

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_np = np.array(img)
        mask_np = np.array(mask)

        mask_bool = mask_np > 0

        background = np.ones_like(img_np) * 255

        result_np = np.where(mask_bool[..., None], img_np, background)

        result_img = Image.fromarray(result_np.astype(np.uint8))
        output_path = os.path.join(output_dir, img_filename)
        result_img.save(output_path)

        print(f"Verarbeitet und gespeichert: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Wendet Segmentationsmasken auf Originalbilder an und füllt Hintergrund weiß. "
                    "Erweiterung für unterschiedliche Extensions (z. B. JPG vs. PNG)."
    )
    parser.add_argument("image_dir", help="Ordner mit den Originalbildern")
    parser.add_argument("mask_dir", help="Ordner mit den Segmentationsmasken")
    parser.add_argument("output_dir", help="Ordner zum Speichern der bearbeiteten Bilder")
    args = parser.parse_args()

    process_folders(args.image_dir, args.mask_dir, args.output_dir)
    print("Fertig! Alle Bilder wurden bearbeitet.")


if __name__ == "__main__":
    main()
