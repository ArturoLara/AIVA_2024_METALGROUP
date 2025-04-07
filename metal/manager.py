from preprocessing import PreprocessingManager, ExamplePreprocessingMethod
from detection import DetectorManager, ExampleDetectionMethod
from tools import Tools

class MainManager:
    def __init__(self, config_path, image_path):
        self.config = Tools.parse_config(config_path)
        self.image_path = image_path

    def start(self):
        # Leer imagen
        image = Tools.read_image(self.image_path)

        # Configurar preprocesadores
        preprocessing_manager = PreprocessingManager()
        for method_name in self.config.get("preprocessing_methods", []):
            if method_name == "ExamplePreprocessingMethod":
                preprocessing_manager.add_method(ExamplePreprocessingMethod())

        # Configurar detector
        detector_manager = DetectorManager()
        method_name = self.config.get("detector_methods", [])
        if method_name == "ExampleDetectionMethod":
            detector_manager.method = ExampleDetectionMethod()

        # Ejecutar preprocesadores
        processed_image = preprocessing_manager.execute_all(image)

        # Ejecutar detección
        results = detector_manager.execute(processed_image)

        # Mostrar resultados (puedes personalizar esto)
        for result in results:
            print(f"Detección: x={result.px}, y={result.py}, w={result.width}, h={result.height}")
