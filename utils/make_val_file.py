import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Erstellt eine test.txt mit Bilddateinamen im angegebenen Verzeichnis"
    )
    parser.add_argument(
        'directory',
        help='Pfad zum Zielverzeichnis, in dem test.txt erstellt wird'
    )
    parser.add_argument(
        'filenames',
        nargs='+',
        help='Liste der Bilddateinamen, z.B. val.jpg'
    )
    args = parser.parse_args()

    # Verzeichnis erstellen, falls es nicht existiert
    os.makedirs(args.directory, exist_ok=True)

    # Pfad zur Ausgabedatei
    output_path = os.path.join(args.directory, 'test.txt')

    # Datei im Schreibmodus öffnen (überschreibt bestehende test.txt)
    with open(output_path, 'w', encoding='utf-8') as f:
        for name in args.filenames:
            f.write(f"{name}\n")

    print(f"test.txt wurde erfolgreich in '{args.directory}' erstellt.")

if __name__ == '__main__':
    main()
