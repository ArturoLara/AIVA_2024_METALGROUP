import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from metal.preprocessing import *
from metal.detection import *
from metal.tools import Tools
from main import MainManager


class TestMainManager(unittest.TestCase):
    @patch('metal.tools.Tools.parse_config')
    def setUp(self, mock_parse_config):
        # Configuración base para todos los tests
        self.mock_config = {
            "defect_type": "auto",
            "other_params": {"example": 123}
        }
        mock_parse_config.return_value = self.mock_config
        self.image_path = "test_image.jpg"
        self.manager = MainManager("dummy_config.json", self.image_path)

    def test_initialization(self):
        """Verifica la inicialización correcta del manager"""
        self.assertEqual(self.manager.image_path, self.image_path)
        self.assertEqual(self.manager.config, self.mock_config)
        self.assertIsNone(self.manager.scratches_manager)
        self.assertIsNone(self.manager.patches_manager)

    @patch('metal.tools.Tools.read_image')
    def test_scratches_processing(self, mock_read_image):
        """Test configuración específica para scratches"""
        # Configurar mocks
        self.mock_config["defect_type"] = "scratches"
        mock_read_image.return_value = np.zeros((100, 100), dtype=np.uint8)

        # Ejecutar flujo principal
        results = self.manager.start()

        # Verificar preprocesadores
        scratch_methods = self.manager.scratches_manager.methods
        self.assertEqual(len(scratch_methods), 3)
        self.assertIsInstance(scratch_methods[0], CLAHEMethod)
        self.assertIsInstance(scratch_methods[1], BrightScratchMethod)
        self.assertIsInstance(scratch_methods[2], MorphologyMethod)

    @patch('metal.tools.Tools.read_image')
    def test_patches_processing(self, mock_read_image):
        """Test configuración específica para patches"""
        self.mock_config["defect_type"] = "patches"
        mock_read_image.return_value = np.zeros((100, 100), dtype=np.uint8)

        results = self.manager.start()

        # Verificar preprocesadores
        patch_methods = self.manager.patches_manager.methods
        self.assertEqual(len(patch_methods), 5)
        self.assertIsInstance(patch_methods[0], GaussianBlurMethod)
        self.assertIsInstance(patch_methods[1], LocalContrastMethod)

    @patch('metal.tools.Tools.read_image')
    def test_auto_processing(self, mock_read_image):
        """Test modo automático con ambos procesadores"""
        mock_read_image.return_value = np.zeros((100, 100), dtype=np.uint8)

        results = self.manager.start()

        # Verificar ambos preprocesadores
        self.assertIsNotNone(self.manager.scratches_manager)
        self.assertIsNotNone(self.manager.patches_manager)

    @patch('metal.tools.Tools.read_image')
    def test_invalid_defect_type(self, mock_read_image):
        """Test tipo de defecto inválido (debería usar auto)"""
        self.mock_config["defect_type"] = "invalid"
        mock_read_image.return_value = np.zeros((100, 100), dtype=np.uint8)

        results = self.manager.start()

    @patch('metal.tools.Tools.read_image')
    def test_full_integration(self, mock_read_image):
        """Test de integración completa con salida simulada"""
        # Configurar mock de imagen y detector
        mock_image = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
        mock_read_image.return_value = mock_image

        # Ejecutar flujo completo
        results = self.manager.start()

        # Verificar formato de resultados
        self.assertIsInstance(results, list)
        if len(results) > 0:
            detection = results[0]
            self.assertTrue(hasattr(detection, 'px'))
            self.assertTrue(hasattr(detection, 'py'))
            self.assertTrue(hasattr(detection, 'width'))
            self.assertTrue(hasattr(detection, 'height'))


if __name__ == '__main__':
    unittest.main()
