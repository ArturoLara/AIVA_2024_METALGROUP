import unittest
import numpy as np
from metal import defect_detector


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


if __name__ == '__main__':
    unittest.main()
