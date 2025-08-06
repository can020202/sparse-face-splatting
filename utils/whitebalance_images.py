import os
import cv2
import numpy as np
from glob import glob

# Pfad zum data-Ordner (relativ zum Skript-Standort)
data_root = 'data'

# Liste aller Unterordner (Personen) in data/
persons = [
    d for d in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, d))
]

def whitebalance_top_center_pixel(image):
    """
    Führt den Weißabgleich anhand des Pixels oben in der Mitte durch.
    """
    h, w, _ = image.shape
    pixel = image[400, w // 2]
    mean_gray = np.mean(pixel)

    scale = mean_gray / pixel
    corrected = image.astype(np.float32)
    for c in range(3):
        corrected[:, :, c] *= scale[c]
    return np.clip(corrected, 0, 255).astype(np.uint8)

def normalize_brightness(image_rgb, target=127.0):
    """
    Skaliert die Helligkeit so, dass die durchschnittliche Luminanz = target ist.
    Luminanz ≈ 0.299 R + 0.587 G + 0.114 B
    """
    img = image_rgb.astype(np.float32)
    lum = img[...,0]*0.299 + img[...,1]*0.587 + img[...,2]*0.114
    mean_lum = np.mean(lum)
    if mean_lum == 0:
        return image_rgb  # Vermeidung Division durch Null
    factor = target / mean_lum
    img *= factor
    return np.clip(img, 0, 255).astype(np.uint8)

for person in persons:
    input_dir = os.path.join(data_root, person, 'original_images')
    output_dir = os.path.join(data_root, person, 'original_images_white_lum')
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.png'))
    print(f"Bearbeite {person} ({len(image_paths)} Bilder) …")

    for path in image_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Konnte Bild nicht laden: {path}")
            continue

        # 1. Weißabgleich
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        wb = whitebalance_top_center_pixel(rgb)

        # 2. Helligkeitsnormierung (Luminanz auf ~127 bringen)
        norm = normalize_brightness(wb, target=150.0)

        # 3. Für bestimmte Posen (front, left, mid_left) zusätzlichen Boost
        fn = os.path.basename(path).lower()
        # Überprüfen, ob im Dateinamen einer der Schlüsselbegriffe vorkommt
        if ('front' in fn) or ('mid_left' in fn) or (('left' in fn) and not 'mid_right' in fn):
            # Faktor >1 = etwas heller. 1.1 heißt +10 %. Kann angepasst werden.
            boost = 1.05
            imgf = norm.astype(np.float32) * boost
            norm = np.clip(imgf, 0, 255).astype(np.uint8)

        # Zurück in BGR und speichern
        out_bgr = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(save_path, out_bgr)

    print(f"{person} abgeschlossen – gespeichert in {output_dir}\n")
