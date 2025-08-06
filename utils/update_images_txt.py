
import argparse
import os
import sys


def update_images_txt(original_file: str,
                      new_names_file: str | None,
                      output_file: str) -> None:
    """Erstellt eine neue COLMAP-images.txt mit (optional) neuen Bildnamen."""

    # ------------------------------------------------------------
    # 1) Neue Bildnamen laden (falls Pfad angegeben)
    # ------------------------------------------------------------
    new_names: list[str] = []
    if new_names_file:                                     # <-- leer/None bedeutet: keine Umbenennung
        try:
            with open(new_names_file, 'r', encoding='utf-8') as nf:
                new_names = [ln.strip() for ln in nf if ln.strip()]
        except FileNotFoundError:
            sys.exit(f"Fehler: Datei '{new_names_file}' nicht gefunden.")

    # ------------------------------------------------------------
    # 2) Originale images.txt einlesen
    # ------------------------------------------------------------
    with open(original_file, 'r', encoding='utf-8') as of:
        lines = of.readlines()

    header, content = [], []
    for ln in lines:
        (header if ln.startswith("#") else content).append(ln.rstrip("\n"))

    if len(content) % 2 != 0:
        sys.exit("Fehler: Die Anzahl der Zeilen im Inhalt ist ungerade – Formatfehler.")

    num_images = len(content) // 2
    if new_names and num_images != len(new_names):
        sys.exit(f"Fehler: {num_images} Bilder in images.txt, "
                 f"aber {len(new_names)} Namen in der Namensliste.")

    # ------------------------------------------------------------
    # 3) Neue images.txt zusammenbauen
    # ------------------------------------------------------------
    updated_content: list[str] = []

    for i in range(num_images):
        param_line = content[2 * i].strip()          # Zeile mit Pose + Dateiname
        tokens     = param_line.split()

        if len(tokens) < 10:
            sys.exit("Fehler: Eine Parameterzeile hat weniger als 10 Einträge.")

        # Dateinamen ersetzen, falls wir eine Liste haben
        if new_names:
            tokens[-1] = new_names[i]

        updated_content.append(" ".join(tokens))
        updated_content.append("")                   # Leerzeile anstelle der Points2D

    # ------------------------------------------------------------
    # 4) Schreiben
    # ------------------------------------------------------------
    with open(output_file, 'w', encoding='utf-8') as out:
        for ln in header:
            out.write(ln + "\n")
        for ln in updated_content:
            out.write(ln + "\n")

    print(f"[OK] Neue images.txt wurde unter '{output_file}' gespeichert.")


# ----------------------------------------------------------------
# CLI-Handling
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Aktualisiert eine COLMAP images.txt mit optional neuen "
                     "Bilddateinamen. Wenn keine Namensliste angegeben ist, "
                     "werden die alten Namen beibehalten.")
    )
    parser.add_argument(
        "--original", required=True,
        help="Pfad zur originalen images.txt (COLMAP-Format)")
    parser.add_argument(
        "--new_names", default=None,
        help="(Optional) TXT-Datei mit neuen Bildnamen – jeder Name in neuer Zeile")
    parser.add_argument(
        "--output", required=True,
        help="Pfad für die erzeugte images.txt")

    args = parser.parse_args()

    update_images_txt(args.original, args.new_names, args.output)