import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from metal.detection import *


class TestDetectionResult(unittest.TestCase):

    def test_detection_result_init(self):
        # Crear instancia
        result = DetectionResult(10, 20, 30, 40)

        # Verificar propiedades
        self.assertEqual(result.px, 10)
        self.assertEqual(result.py, 20)
        self.assertEqual(result.width, 30)
        self.assertEqual(result.height, 40)

    def test_detection_result_iter(self):
        # Crear instancia
        result = DetectionResult(10, 20, 30, 40)

        # Verificar que se puede iterar
        x, y, w, h = result
        self.assertEqual(x, 10)
        self.assertEqual(y, 20)
        self.assertEqual(w, 30)
        self.assertEqual(h, 40)

class TestDetectorManager(unittest.TestCase):

    def test_execute(self):
        # Crear método mock
        method = MagicMock()
        expected_results = [DetectionResult(10, 20, 30, 40)]
        method.detect.return_value = expected_results

        # Crear manager
        manager = DetectorManager(method)

        # Crear imagen de prueba
        test_image = np.zeros((100, 100), dtype=np.uint8)

        # Ejecutar manager
        results = manager.execute(test_image)

        # Verificar que se llamó al método detect
        method.detect.assert_called_once_with(test_image)

        # Verificar que los resultados son los esperados
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].px, 10)
        self.assertEqual(results[0].py, 20)
        self.assertEqual(results[0].width, 30)
        self.assertEqual(results[0].height, 40)


if __name__ == '__main__':
    unittest.main()
