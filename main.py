import argparse
from metal.manager import MainManager
#from metal.detection import DetectionResult
import cv2

def dibujar_rectangulos_y_guardar(imagen_path, objetos, salida_path):
    """
    Dibuja rectángulos rojos sobre una imagen basada en una lista de objetos y guarda la imagen resultante.

    :param imagen_path: Ruta de la imagen original.
    :param objetos: Lista de objetos con atributos px, py, height y weight.
    :param salida_path: Ruta donde se guardará la imagen resultante.
    """
    # Cargar la imagen
    imagen = cv2.imread(imagen_path)

    # Dibujar los rectángulos
    for obj in objetos:
        # Coordenadas del rectángulo (esquina superior izquierda y esquina inferior derecha)
        esquina_superior_izquierda = (obj.px, obj.py)
        esquina_inferior_derecha = (obj.px + obj.width, obj.py + obj.height)

        # Dibujar el rectángulo rojo (color en BGR: (0, 0, 255))
        cv2.rectangle(imagen, esquina_superior_izquierda, esquina_inferior_derecha, (0, 0, 255), 2)

    # Guardar la imagen resultante
    cv2.imwrite(salida_path, imagen)
    print(f"Imagen guardada en: {salida_path}")


def main():
    parser = argparse.ArgumentParser(description="Sistema de análisis de imágenes para detectar imperfecciones.")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración JSON.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen a analizar.")

    args = parser.parse_args()

    manager = MainManager(config_path=args.config, image_path=args.image)
    detections = manager.start()

    dibujar_rectangulos_y_guardar(args.image, detections, "output.jpg")


if __name__ == "__main__":
    main()