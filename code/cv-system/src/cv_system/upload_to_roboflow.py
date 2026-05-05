"""
Sube imágenes anotadas a Roboflow.

Uso:
    upload-roboflow                           # Sube desde images/annotated
    upload-roboflow --input datasets/nuevo    # Sube desde carpeta específica
    upload-roboflow --batch 50                # Sube en lotes de 50
    upload-roboflow --generate-yaml           # Solo genera data.yaml sin subir

Estructura esperada:
    input_dir/
        images/
            predict_000000.jpg
        labels/
            predict_000000.txt   # formato YOLO
        data.yaml                # generado automáticamente
"""
import os
import argparse
from pathlib import Path

from dotenv import load_dotenv

UPLOAD_INPUT_DIR = os.getenv("UPLOAD_INPUT_DIR", "images/annotated")

# Clases del modelo actual (de Cards-yolo11segun/data.yaml)
CLASS_NAMES = [
    'add', 'apple', 'ascending', 'burger', 'descending', 'division',
    'eight', 'filter', 'five', 'four', 'grapes', 'green', 'lg_circle',
    'lg_square', 'md_circle', 'md_square', 'multiply', 'nine', 'one',
    'orange', 'pear', 'purple', 'red', 'result', 'seven', 'six',
    'sm_circle', 'sm_square', 'subtract', 'three', 'two', 'zero'
]


def generate_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """Genera data.yaml compatible con YOLO/Roboflow."""
    yaml_path = output_dir / "data.yaml"

    content = f"""train: images
val: images
test: images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path.write_text(content)
    return yaml_path


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Sube imágenes anotadas a Roboflow")
    parser.add_argument(
        "--input", type=str, default=UPLOAD_INPUT_DIR, help="Carpeta con images/ y labels/"
    )
    parser.add_argument(
        "--batch", type=int, default=100, help="Tamaño del lote"
    )
    parser.add_argument(
        "--project", type=str, default=None, help="Nombre del proyecto en Roboflow"
    )
    parser.add_argument(
        "--workspace", type=str, default=None, help="Workspace en Roboflow"
    )
    parser.add_argument(
        "--generate-yaml", action="store_true", help="Solo genera data.yaml sin subir"
    )
    args = parser.parse_args()

    # Verificar carpetas
    input_dir = Path(args.input)
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists():
        print(f"ERROR: No existe {images_dir}")
        return 1

    if not labels_dir.exists():
        print(f"ERROR: No existe {labels_dir}")
        return 1

    # Generar data.yaml
    yaml_path = generate_data_yaml(input_dir, CLASS_NAMES)
    print(f"Generado: {yaml_path}")

    if args.generate_yaml:
        print("Modo --generate-yaml: solo se generó data.yaml")
        return 0

    # Verificar API key para subir
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY no está configurada en .env")
        print("Obtén tu API key en: https://app.roboflow.com/settings/api")
        return 1

    # Importar roboflow solo si vamos a subir
    from roboflow import Roboflow

    # Obtener lista de imágenes
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        print(f"No hay imágenes en {images_dir}")
        return 1

    print(f"Encontradas {len(images)} imágenes")

    # Conectar a Roboflow
    rf = Roboflow(api_key=api_key)

    workspace = args.workspace or os.getenv("ROBOFLOW_WORKSPACE", "thesis-s4jik")
    project_name = args.project or os.getenv("ROBOFLOW_PROJECT", "thesis-s4jik")

    print(f"Conectando a {workspace}/{project_name}...")
    project = rf.workspace(workspace).project(project_name)

    # Subir imágenes con anotaciones
    uploaded = 0
    skipped = 0
    errors = 0

    for i, img_path in enumerate(images):
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"[SKIP] Sin anotación: {img_path.name}")
            skipped += 1
            continue

        try:
            # Roboflow acepta YOLO format directamente
            project.upload(
                image_path=str(img_path),
                annotation_path=str(label_path),
                annotation_format="yolov8",
            )
            uploaded += 1
            print(f"[{uploaded}/{len(images)}] Subido: {img_path.name}")

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            errors += 1

        # Mostrar progreso cada batch
        if (i + 1) % args.batch == 0:
            print(f"--- Progreso: {i + 1}/{len(images)} ---")

    print(f"\nResumen:")
    print(f"  Subidas: {uploaded}")
    print(f"  Omitidas: {skipped}")
    print(f"  Errores: {errors}")

    return 0


if __name__ == "__main__":
    exit(main())
