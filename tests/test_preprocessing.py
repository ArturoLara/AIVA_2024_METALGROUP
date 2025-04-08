import unittest
import numpy as np
from metal.preprocessing import ExamplePreprocessingMethod, UmbralizeMethod, CannyMethod, PreprocessingManager


class TestPreprocessingMethods(unittest.TestCase):
    def setUp(self):
        # Crear una imagen de prueba (200x200 con valores constantes de 100)
        self.image = np.ones((200, 200), dtype=np.uint8) * 100

    def test_example_preprocessing_method(self):
        method = ExamplePreprocessingMethod()
        processed_image = method.process(self.image)

        # Verificar que la imagen procesada tiene las dimensiones esperadas
        self.assertEqual(processed_image.shape, (200, 200))

        # Verificar que la imagen procesada es completamente negra (valores 0)
        self.assertTrue(np.array_equal(processed_image, np.zeros((200, 200), dtype=np.uint8)))

    def test_umbralize_method(self):
        method = UmbralizeMethod()
        processed_image = method.process(self.image)

        # Verificar que la imagen procesada tiene las dimensiones esperadas
        self.assertEqual(processed_image.shape, (200, 200))

        # Verificar que los píxeles menores o iguales a 150 son negros (0)
        self.assertTrue(np.all(processed_image[self.image <= 150] == 0))

        # Verificar que los píxeles mayores a 150 son blancos (255)
        self.assertTrue(np.all(processed_image[self.image > 150] == 255))

    def test_canny_method(self):
        method = CannyMethod()
        processed_image = method.process(self.image)

        # Verificar que la imagen procesada tiene las dimensiones esperadas
        self.assertEqual(processed_image.shape, (200, 200))

    def test_preprocessing_manager(self):
        manager = PreprocessingManager()

        # Añadir los métodos de preprocesamiento al gestor
        manager.add_method(ExamplePreprocessingMethod())
        manager.add_method(UmbralizeMethod())
        manager.add_method(CannyMethod())

        # Ejecutar todos los métodos de preprocesamiento en la imagen de prueba
        processed_image = manager.execute_all(self.image)

        # Verificar que la imagen final procesada tiene las dimensiones esperadas
        self.assertEqual(processed_image.shape, (200, 200))


if __name__ == "__main__":
    unittest.main()
