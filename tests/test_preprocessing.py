import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from metal.preprocessing import *


class TestPreprocessingMethods(unittest.TestCase):

    def setUp(self):
        # Crear una imagen de prueba simple
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        # Añadir algunos elementos para que sea más realista
        self.test_image[40:60, 40:60] = 255  # Un cuadrado blanco en el centro

    def test_gaussian_blur_method(self):
        # Crear instancia
        method = GaussianBlurMethod(sigma=1.5)

        # Aplicar método
        result = method.process(self.test_image)

        # Verificar que la imagen ha sido procesada
        self.assertEqual(result.shape, self.test_image.shape)
        # El desenfoque debería suavizar los bordes
        self.assertTrue(np.any(result[38, 38:42] > 0))

    def test_median_blur_method(self):
        # Crear instancia con valor impar para ksize
        method = MedianBlurMethod(ksize=3)

        # Añadir ruido a la imagen
        noisy_image = self.test_image.copy()
        noisy_image[45, 45] = 0  # Un píxel negro en el cuadrado blanco

        # Aplicar método
        result = method.process(noisy_image)

        # Verificar que el ruido ha sido eliminado
        self.assertEqual(result[45, 45], 255)

    def test_threshold_method(self):
        # Crear instancia
        method = ThresholdMethod(factor=0.5)

        # Crear imagen con valores conocidos
        gradient_image = np.linspace(0, 255, 100 * 100, dtype=np.uint8).reshape(100, 100)

        # Aplicar método
        result = method.process(gradient_image)

        # Verificar que la umbralización se ha aplicado correctamente
        threshold = np.max(gradient_image) * 0.5
        expected = (gradient_image > threshold).astype(np.uint8) * 255
        np.testing.assert_array_equal(result, expected)

    def test_morphology_method_close(self):
        # Crear instancia para operación de cierre
        method = MorphologyMethod(operation='close', kernel_size=(3, 3))

        # Crear imagen con dos cuadrados cercanos
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[40:45, 40:45] = 255  # Primer cuadrado
        test_image[46:51, 40:45] = 255  # Segundo cuadrado

        # Aplicar método
        result = method.process(test_image)

        # Verificar que se aplicó la operación correcta
        self.assertNotEqual(np.sum(result[45, 40:45]), 0)  # Debería haber llenado el espacio

    def test_clahe_method(self):
        # Crear instancia
        method = CLAHEMethod(clip_limit=2.0, grid_size=(8, 8))

        # Crear imagen con bajo contraste
        low_contrast_image = np.ones((100, 100), dtype=np.uint8) * 50
        low_contrast_image[40:60, 40:60] = 100

        # Mockear cv2.createCLAHE
        with patch('cv2.createCLAHE') as mock_create_clahe:
            mock_clahe = MagicMock()
            mock_clahe.apply.return_value = np.ones((100, 100), dtype=np.uint8) * 150
            mock_create_clahe.return_value = mock_clahe

            # Aplicar método
            result = method.process(low_contrast_image)

            # Verificar que se llamó a createCLAHE con los parámetros correctos
            mock_create_clahe.assert_called_once_with(clipLimit=2.0, tileGridSize=(8, 8))
            mock_clahe.apply.assert_called_once()
            self.assertTrue(np.all(result == 150))

    def test_preprocessing_manager(self):
        # Crear manager
        manager = PreprocessingManager()

        # Crear métodos mock
        method1 = MagicMock()
        method1.process.return_value = self.test_image.copy() + 50

        method2 = MagicMock()
        method2.process.return_value = self.test_image.copy() + 100

        # Añadir métodos
        manager.add_method(method1)
        manager.add_method(method2)

        # Ejecutar todos los métodos
        result = manager.execute_all(self.test_image)

        # Verificar que se llamaron los métodos en orden
        method1.process.assert_called_once()
        method2.process.assert_called_once()

        # Verificar que el resultado es el esperado (la salida del último método)
        expected = self.test_image.copy() + 100
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
