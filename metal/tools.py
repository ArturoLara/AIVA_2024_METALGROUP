import importlib

import cv2
import numpy as np
import json

class Tools:
    @staticmethod
    def read_image(file_path):
        return cv2.imread(file_path)

    @staticmethod
    def parse_config(config_path):
        with open(config_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def create_instance(class_name, params, module_name=None):
        try:
            if module_name is not None:
                module = importlib.import_module(module_name)
            else:
                import sys
                module = sys.modules[__name__]
            class_ = getattr(module, class_name)
            # Procesamiento de parámetros igual que antes...
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, list) and key in ["grid_size", "kernel_size"]:
                    processed_params[key] = tuple(value)
                else:
                    processed_params[key] = value
            return class_(**processed_params)
        except Exception as e:
            print(f"Error creando instancia de {class_name} en {module_name}: {e}")
            raise
