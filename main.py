import argparse
from metal.manager import MainManager


def main():
    parser = argparse.ArgumentParser(description="Sistema de análisis de imágenes para detectar imperfecciones.")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración JSON.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen a analizar.")

    args = parser.parse_args()

    manager = MainManager(config_path=args.config, image_path=args.image)
    manager.start()


if __name__ == "__main__":
    main()