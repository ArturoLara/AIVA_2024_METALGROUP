import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from metal.preprocessing import PreprocessingManager, UmbralizeMethod, CannyMethod
from metal.detection import DetectorManager, ContrastMethod
from metal.tools import Tools
from metal.manager import MainManager


class TestMainManager(unittest.TestCase):
    def setUp(self):
        # Crear configuraciones de prueba
        self.config_path = "config.yaml"
        self.image_path = "test_image.jpg"

        # Mock de la configuración
        self.mock_config = {
            "preprocessing_methods": ["ExamplePreprocessingMethod", "CannyMethod"],
            "detector_methods": "ExampleDetectionMethod"
        }

        # Mock de una imagen de prueba (200x200)
        self.mock_image = np.ones((200, 200), dtype=np.uint8) * 100

    @patch("metal.tools.Tools.parse_config")
    @patch("metal.tools.Tools.read_image")
    @patch("metal.preprocessing.PreprocessingManager.execute_all")
    @patch("metal.detection.DetectorManager.execute")
    def test_main_manager_workflow(self, mock_execute_detection, mock_execute_preprocessing, mock_read_image,
                                   mock_parse_config):
        # Configurar los mocks
        mock_parse_config.return_value = self.mock_config
        mock_read_image.return_value = self.mock_image

        # Mock del resultado del preprocesamiento (imagen procesada)
        processed_image = np.zeros((200, 200), dtype=np.uint8)
        mock_execute_preprocessing.return_value = processed_image

        # Mock del resultado de la detección
        detection_results = [MagicMock(px=10, py=10, width=50, height=50)]
        mock_execute_detection.return_value = detection_results

        # Instanciar MainManager y ejecutar el flujo principal
        manager = MainManager(self.config_path, self.image_path)
        manager.start()

        # Verificar que se llamó a parse_config con la ruta correcta
        mock_parse_config.assert_called_once_with(self.config_path)

        # Verificar que se llamó a read_image con la ruta correcta
        mock_read_image.assert_called_once_with(self.image_path)

        # Verificar que se ejecutaron los preprocesadores y el detector
        mock_execute_preprocessing.assert_called_once_with(self.mock_image)
        mock_execute_detection.assert_called_once_with(processed_image)

    @patch("metal.tools.Tools.parse_config")
    @patch("metal.tools.Tools.read_image")
    def test_main_manager_invalid_detector_method(self, mock_read_image, mock_parse_config):
        # Configuración con un método de detección inválido
        invalid_config = {
            "preprocessing_methods": ["ExamplePreprocessingMethod"],
            "detector_methods": "InvalidDetectionMethod"
        }

        mock_parse_config.return_value = invalid_config
        mock_read_image.return_value = self.mock_image

        manager = MainManager(self.config_path, self.image_path)

        with self.assertRaises(ValueError) as context:
            manager.start()

        self.assertEqual(str(context.exception), "Método de detección desconocido: InvalidDetectionMethod")


if __name__ == "__main__":
    unittest.main()
