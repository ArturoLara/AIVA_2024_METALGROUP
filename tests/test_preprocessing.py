import unittest
import numpy as np
from metal import preprocessing

class TestImageProcessing(unittest.TestCase):

    def test_process_image_returns_correct_shape(self):
        # Se utiliza una ruta dummy ya que en el mockup no se carga realmente la imagen.
        processed_image = preprocessing.process_image("dummy_path.jpg")
        self.assertIsInstance(processed_image, np.ndarray)
        self.assertEqual(processed_image.shape, (200, 200))
        self.assertEqual(processed_image.dtype, 'uint8')


if __name__ == '__main__':
    unittest.main()
