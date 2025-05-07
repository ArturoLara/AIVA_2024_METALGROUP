import unittest
from unittest.mock import patch, mock_open
import numpy as np
import json
from metal.tools import Tools


class TestTools(unittest.TestCase):

    @patch('cv2.imread')
    def test_read_image(self, mock_imread):
        # Configurar el mock
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        # Ejecutar el método
        result = Tools.read_image('fake/path/image.jpg')

        # Verificar que se llamó cv2.imread con los parámetros correctos
        mock_imread.assert_called_once_with('fake/path/image.jpg')

        # Verificar que el resultado es el esperado
        self.assertEqual(result.shape, mock_image.shape)
        self.assertEqual(result.dtype, mock_image.dtype)

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_parse_config(self, mock_json_load, mock_file_open):
        # Configurar el mock
        mock_config = {'defect_type': 'scratches'}
        mock_json_load.return_value = mock_config

        # Ejecutar el método
        result = Tools.parse_config('fake/path/config.json')

        # Verificar que se abrió el archivo correcto
        mock_file_open.assert_called_with('fake/path/config.json', 'r')

        # Verificar que se llamó a json.load
        mock_json_load.assert_called_once()

        # Verificar que el resultado es el esperado
        self.assertEqual(result, mock_config)

    def test_create_instance(self):
        # Crear una instancia de una clase conocida con parámetros
        params = {'sigma': 1.5}
        from metal.preprocessing import GaussianBlurMethod

        instance = Tools.create_instance('GaussianBlurMethod', params, 'metal.preprocessing')

        # Verificar que la instancia es del tipo correcto
        self.assertIsInstance(instance, GaussianBlurMethod)

        # Verificar que los parámetros se pasaron correctamente
        self.assertEqual(instance.sigma, 1.5)

    def test_create_instance_with_tuple_params(self):
        # Probar la conversión de listas a tuplas
        params = {'grid_size': [8, 8], 'clip_limit': 2.5}
        from metal.preprocessing import CLAHEMethod

        instance = Tools.create_instance('CLAHEMethod', params, 'metal.preprocessing')

        # Verificar que la instancia es del tipo correcto
        self.assertIsInstance(instance, CLAHEMethod)

        # Verificar que los parámetros se pasaron correctamente y la lista se convirtió a tupla
        self.assertEqual(instance.grid_size, (8, 8))
        self.assertEqual(instance.clip_limit, 2.5)

    def test_create_instance_error(self):
        # Probar el manejo de errores
        params = {}

        # Debería lanzar una excepción al intentar crear una instancia de una clase que no existe
        with self.assertRaises(Exception):
            Tools.create_instance('NonExistentClass', params, 'metal.preprocessing')


if __name__ == '__main__':
    unittest.main()
