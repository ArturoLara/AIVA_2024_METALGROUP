import unittest
import cv2
import numpy as np
from metal.detection import DetectorManager, ContrastMethod, DetectionResult


class TestDefectDetector(unittest.TestCase):
    def test_bounding_box_within_image(self):
        """
        Verifica que para cada bounding box (x, y, w, h) detectado,
        se cumpla que:
          - x y y sean mayores o iguales que 0, y
          - x+w y y+h no excedan las dimensiones 200x200.
        """
        # Utilizamos una imagen dummy no nula para simular una detección real.
        dummy_image = np.ones((200, 200), dtype=np.uint8) * 255
        defects = DetectorManager(ContrastMethod()).execute(dummy_image)
        for defect in defects:
            x, y, w, h = defect
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertGreaterEqual(w, 0)
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(x + w, 200)
            self.assertLessEqual(y + h, 200)

    def test_max_defects_count(self):
        """
        Comprueba que la lista de defectos detectados contenga al menos 1 elemento
        y no exceda 5 elementos, cumpliendo los límites establecidos en el SRS.
        """
        # Para este test, se utiliza una imagen dummy no nula.
        dummy_image = np.ones((200, 200), dtype=np.uint8) * 255
        defects = DetectorManager(ExampleDetectionMethod()).execute(dummy_image)
        self.assertGreaterEqual(len(defects), 1)
        self.assertLessEqual(len(defects), 5)

    def setUp(self):
        # Crear imágenes de prueba
        self.black_image = np.zeros((200, 200), dtype=np.uint8)  # Imagen completamente negra
        self.white_image = np.ones((200, 200), dtype=np.uint8) * 255  # Imagen completamente blanca

        # Imagen con un rectángulo blanco en el centro
        self.image_with_rectangle = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(self.image_with_rectangle, (50, 50), (150, 150), (255), -1)

    def test_contrast_method_no_contours(self):
        method = ContrastMethod()
        empty_result = DetectionResult(0, 0, 0, 0)  # Simular un resultado vacío

        # Detectar en una imagen negra (sin contornos)
        result = method.detect(self.black_image)

        # Verificar que no se detectan contornos
        self.assertEqual(result.px, empty_result.px)
        self.assertEqual(result.py, empty_result.py)
        self.assertEqual(result.height, empty_result.height)
        self.assertEqual(result.width, empty_result.width)

    def test_contrast_method_with_contours(self):
        method = ContrastMethod()

        # Detectar en una imagen con un rectángulo blanco
        result = method.detect(self.image_with_rectangle)

        # Verificar que se detecta el rectángulo blanco correctamente
        self.assertEqual(len(result), 1)  # Solo un contorno principal
        zona = result[0]

        # Verificar las coordenadas del rectángulo detectado
        self.assertEqual((zona.px, zona.py), (50, 50))
        self.assertEqual((zona.width, zona.height), (101, 101))  # Incluye bordes del rectángulo

    def test_detector_manager_with_contrast_method(self):
        manager = DetectorManager(method=ContrastMethod())

        # Ejecutar detección con el método de contraste en una imagen con un rectángulo blanco
        result = manager.execute(self.image_with_rectangle)

        # Verificar que se detecta el rectángulo correctamente
        self.assertEqual(len(result), 1)  # Solo un contorno principal detectado


if __name__ == '__main__':
    unittest.main()
