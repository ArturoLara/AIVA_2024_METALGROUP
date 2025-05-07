import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from metal.manager import MainManager
from metal.preprocessing import PreprocessingManager
from metal.detection import DetectorManager


class TestMainManager(unittest.TestCase):

    def setUp(self):
        # Configurar paths
        self.config_path = 'fake/path/config.json'
        self.image_path = 'fake/path/image.jpg'

        # Crear manager
        self.manager = MainManager(self.config_path, self.image_path)

        # Crear imagen de prueba
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    @patch('metal.tools.Tools.read_image')
    @patch('metal.tools.Tools.parse_config')
    @patch('metal.manager.MainManager._init_preprocessing_manager')
    @patch('metal.manager.MainManager._init_detector')
    def test_start_with_config(self, mock_init_detector, mock_init_preprocessing, mock_parse_config, mock_read_image):
        # Configurar mocks
        mock_read_image.return_value = self.test_image
        mock_config = {'defect_type': 'scratches'}
        mock_parse_config.return_value = mock_config

        # Configurar manager
        self.manager.scratches_manager = MagicMock()
        self.manager.scratches_manager.execute_all.return_value = self.test_image

        self.manager.detector_manager = MagicMock()
        expected_results = [MagicMock()]
        self.manager.detector_manager.execute.return_value = expected_results

        # Ejecutar start
        results = self.manager.start()

        # Verificar que se leyó la imagen
        mock_read_image.assert_called_once_with(self.image_path)

        # Verificar que se parseó la configuración
        mock_parse_config.assert_called_once_with(self.config_path)

        # Verificar que se inicializaron los managers correctos
        mock_init_preprocessing.assert_called_once_with('scratches')
        mock_init_detector.assert_called_once_with('scratches')

        # Verificar que se ejecutaron los métodos de preprocesamiento
        self.manager.scratches_manager.execute_all.assert_called_once()

        # Verificar que se ejecutó la detección
        self.manager.detector_manager.execute.assert_called_once()

        # Verificar que se devolvieron los resultados esperados
        self.assertEqual(results, expected_results)

if __name__ == '__main__':
    unittest.main()
