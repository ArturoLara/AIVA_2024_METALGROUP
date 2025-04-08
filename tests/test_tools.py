import json
import unittest
from unittest.mock import patch, mock_open
import numpy as np
from metal.tools import Tools

class TestTools(unittest.TestCase):
    @patch("cv2.imread")
    def test_read_image(self, mock_imread):
        # Configurar el mock para que devuelva una imagen simulada
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # Imagen blanca
        mock_imread.return_value = mock_image

        # Probar el método read_image
        file_path = "test_image.jpg"
        result = Tools.read_image(file_path)

        # Verificar que cv2.imread fue llamado con el archivo correcto
        mock_imread.assert_called_once_with(file_path)

        # Verificar que el resultado es la imagen simulada
        self.assertTrue(np.array_equal(result, mock_image))

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_parse_config(self, mock_file):
        # Probar el método parse_config
        config_path = "config.json"
        result = Tools.parse_config(config_path)

        # Verificar que se abrió el archivo correcto
        mock_file.assert_called_once_with(config_path, 'r')

        # Verificar que el resultado es un diccionario con los datos esperados
        self.assertEqual(result, {"key": "value"})

    @patch("builtins.open", new_callable=mock_open)
    def test_parse_config_invalid_json(self, mock_file):
        # Configurar el mock para devolver un JSON inválido
        mock_file.return_value.read.return_value = "{invalid_json}"

        # Probar que se lanza una excepción al intentar parsear un JSON inválido
        config_path = "invalid_config.json"
        with self.assertRaises(json.JSONDecodeError):
            Tools.parse_config(config_path)

if __name__ == "__main__":
    unittest.main()
