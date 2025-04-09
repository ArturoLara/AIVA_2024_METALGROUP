import numpy as np
from abc import ABC, abstractmethod
import cv2

class DetectionResult:
    def __init__(self, px, py, width, height):
        self.px = px
        self.py = py
        self.width = width
        self.height = height

    def __iter__(self):
        return iter((self.px, self.py, self.width, self.height))

class DetectionMethod(ABC):

    @abstractmethod
    def detect(self, image):
        """Método que debe implementar el algoritmo de detección"""
        pass

class ContrastMethod(DetectionMethod):
    def detect(self, image):
        # Encontrar los contornos en la imagen de bordes
        contornos, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lista para almacenar las zonas detectadas como objetos DetectionResult
        zonas_detectadas = []

        for contorno in contornos:
            # Calcular el rectángulo delimitador para cada contorno
            x, y, w, h = cv2.boundingRect(contorno)
            zonas_detectadas.append(DetectionResult(x, y, w, h))

        # Ordenar las zonas detectadas por área en orden descendente
        zonas_detectadas.sort(key=lambda zona: zona.width * zona.height, reverse=True)

        # Filtrar zonas que se superponen
        zonas_filtradas = []
        for i, zona in enumerate(zonas_detectadas):
            x1, y1, w1, h1 = zona.px, zona.py, zona.width, zona.height
            area1 = w1 * h1

            superpuesto = False
            for j in range(i):
                x2, y2, w2, h2 = zonas_detectadas[j].px, zonas_detectadas[j].py, zonas_detectadas[j].width, \
                zonas_detectadas[j].height
                area2 = w2 * h2

                # Calcular intersección entre las dos áreas
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    if inter_area / area1 > 0.7 or inter_area / area2 > 0.7:
                        superpuesto = True
                        break

            if not superpuesto:
                zonas_filtradas.append(zona)

            if len(zonas_filtradas) == 5:
                break

        return DetectionResult(0, 0, 0, 0) if len(zonas_filtradas) == 0 else zonas_filtradas

class DetectorManager:
    def __init__(self, method: DetectionMethod):
        self.method = method

    def execute(self, image):
        return self.method.detect(image)
