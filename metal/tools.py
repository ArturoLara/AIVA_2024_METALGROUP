import cv2
import numpy as np
import json

class Tools:
    @staticmethod
    def read_image(file_path):
        """Lee una imagen desde el disco y la convierte en un array de numpy"""
        print(f"Leyendo imagen desde {file_path}")
        return cv2.imread(file_path)

    @staticmethod
    def parse_config(config_path):
        """Lee y parsea un archivo JSON de configuración"""
        print(f"Leyendo configuración desde {config_path}")
        with open(config_path, 'r') as file:
            return json.load(file)
