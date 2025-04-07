import numpy as np
from abc import ABC, abstractmethod

class DetectionResult:
    def __init__(self, px, py, width, height):
        self.px = px
        self.py = py
        self.width = width
        self.height = height

class DetectionMethod(ABC):

    @abstractmethod
    def detect(self, image):
        """Método que debe implementar el algoritmo de detección"""
        pass

class ExampleDetectionMethod(DetectionMethod):
    def detect(self, image):
        """
        Función mockup para detectar defectos en una imagen procesada.

        Si la imagen es la dummy (todos sus píxeles en cero),
        se asume que no hay defectos y se retorna [(0,0,0,0)].

        En otro caso se simula la detección de un defecto con coordenadas dummy.
        """
        print("Detectando imperfecciones en la imagen")

        # Si la imagen es completamente negra, se interpreta como sin defectos.
        if np.sum(image) == 0:
            return [(0, 0, 0, 0)]
        else:
            # Se simula la detección de un defecto con coordenadas fijas.
            return [(10, 10, 50, 50)]

class DetectorManager:
    def __init__(self, method: DetectionMethod):
        self.method = method

    def execute(self, image):
        return self.method.detect(image)
