import cv2
import numpy as np
from abc import ABC, abstractmethod

class PreprocessingMethod(ABC):
    @abstractmethod
    def process(self, image):
        """Método abstracto que debe implementar cada método de preprocesado"""
        pass

class ExamplePreprocessingMethod(PreprocessingMethod):
    def process(self, image):
        """Ejemplo de implementación de un método de preprocesado"""
        print("Procesando imagen con ExamplePreprocessingMethod")
        processed_image = np.zeros((200, 200), dtype=np.uint8)
        return processed_image

class PreprocessingManager:
    def __init__(self):
        self.methods = []

    def add_method(self, method: PreprocessingMethod):
        self.methods.append(method)

    def execute_all(self, image):
        for method in self.methods:
            image = method.process(image)
        return image
