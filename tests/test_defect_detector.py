import unittest
import numpy as np
from aiva import defect_detector


class TestDefectDetector(unittest.TestCase):

    def test_detect_defects_no_defects(self):
        # Se crea una imagen dummy compuesta solo por ceros
        dummy_image = np.zeros((200, 200), dtype=np.uint8)
        defects = defect_detector.detect_defects(dummy_image)
        # Debe retornar [(0,0,0,0)] indicando que no se detectaron defectos.
        self.assertEqual(defects, [(0, 0, 0, 0)])

    def test_detect_defects_with_defect(self):
        # Se crea una imagen dummy no nula para simular la detección de defectos
        dummy_image = np.ones((200, 200), dtype=np.uint8) * 255
        defects = defect_detector.detect_defects(dummy_image)
        # Se espera que se detecte un defecto dummy con coordenadas fijas.
        self.assertIsInstance(defects, list)
        self.assertNotEqual(defects, [(0, 0, 0, 0)])
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0], (10, 10, 50, 50))

    def test_bounding_box_within_image(self):
        """
        Verifica que para cada bounding box (x, y, w, h) detectado,
        se cumpla que:
          - x y y sean mayores o iguales que 0, y
          - x+w y y+h no excedan las dimensiones 200x200.
        """
        # Utilizamos una imagen dummy no nula para simular una detección real.
        dummy_image = np.ones((200, 200), dtype=np.uint8) * 255
        defects = defect_detector.detect_defects(dummy_image)
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
        defects = defect_detector.detect_defects(dummy_image)
        self.assertGreaterEqual(len(defects), 1)
        self.assertLessEqual(len(defects), 5)


if __name__ == '__main__':
    unittest.main()
