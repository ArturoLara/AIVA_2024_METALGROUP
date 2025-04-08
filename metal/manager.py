from metal.preprocessing import PreprocessingManager, ExamplePreprocessingMethod, UmbralizeMethod, CannyMethod
from metal.detection import DetectorManager, ExampleDetectionMethod, ContrastMethod
from metal.tools import Tools

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
            elif method_name == "UmbralizeMethod":
                preprocessing_manager.add_method(UmbralizeMethod())
            elif method_name == "CannyMethod":
                preprocessing_manager.add_method(CannyMethod())

        # Configurar detector
        detector_manager = None
        method_name = self.config.get("detector_methods", [])
        if method_name == "ExampleDetectionMethod":
            detector_manager = DetectorManager(ExampleDetectionMethod())
        elif method_name == "ContrastMethod":
            detector_manager = DetectorManager(ContrastMethod())
        else:
            raise ValueError(f"Método de detección desconocido: {method_name}")

        # Ejecutar preprocesadores
        processed_image = preprocessing_manager.execute_all(image)

        # Ejecutar detección
        results = detector_manager.execute(processed_image)

        # Mostrar resultados (puedes personalizar esto)
        for result in results:
            print(f"Detección: x={result.px}, y={result.py}, w={result.width}, h={result.height}")

        return results
