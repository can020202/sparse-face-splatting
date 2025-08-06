import subprocess
import os
import sys
import argparse

def run_model_converter(input_path: str, output_path: str):
    # Erstelle den Ausgabeordner, falls er nicht existiert
    os.makedirs(output_path, exist_ok=True)

    # Baue den COLMAP-Befehl als Liste
    cmd = [
        "colmap", "model_converter",
        "--input_path", input_path,
        "--output_path", output_path,
        "--output_type", "TXT"
    ]

    print("Running command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("Model conversion succeeded.")
    except subprocess.CalledProcessError as e:
        print("Error during model conversion:", e)
        sys.exit(e.returncode)

    # Leere points3D.txt komplett, damit keine alten Einträge verbleiben
    points_txt = os.path.join(output_path, "points3D.txt")
    try:
        with open(points_txt, "w") as f:
            # Datei wird geöffnet und sofort geschlossen, wodurch sie geleert wird
            pass
        print(f"Cleared contents of {points_txt}.")
    except Exception as e:
        print(f"Failed to clear {points_txt}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP binary model to TXT with model_converter."
    )
    parser.add_argument(
        "-i", "--input_path",
        required=True,
        help="Pfad zum Ordner mit der COLMAP-Binary-Modelldatei (z.B. cameras.bin, images.bin, points3D.bin)"
    )
    parser.add_argument(
        "-o", "--output_path",
        required=True,
        help="Zielordner für die TXT-Ausgabe"
    )
    args = parser.parse_args()

    run_model_converter(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
