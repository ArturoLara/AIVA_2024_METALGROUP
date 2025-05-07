from metal.preprocessing import *
from metal.detection import *
from metal.tools import Tools
import logging

class MainManager:
    def __init__(self, config_path, image_path):
        self.config_path = config_path
        self.image_path = image_path
        self.config = None
        self.scratches_manager = None
        self.patches_manager = None
        self.detector_manager = None
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)

    def start(self):
        # Leer imagen
        image = Tools.read_image(self.image_path)

        # Configurar preprocesadores
        if self.config_path:
            try:
                self.config = Tools.parse_config(self.config_path)
                self.logger.info(f"Configuración cargada desde {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error cargando configuración: {e}")
                self.config = {}
        else:
            self.config = {}

        # Determinar tipo de defecto a detectar
        defect_type = self.config.get("defect_type", "auto")

        # Inicializar managers para cada tipo de defecto
        if defect_type in ["scratches", "auto"]:
            self._init_preprocessing_manager("scratches")

        if defect_type in ["patches", "auto"]:
            self._init_preprocessing_manager("patches")

        # Inicializar detectores según el tipo de defecto
        if defect_type == "scratches" or defect_type == "auto":
            self._init_detector("scratches")

        if defect_type == "patches" or defect_type == "auto":
            self._init_detector("patches")

        # Ejecutar preprocesadores
        if self.scratches_manager:
            image = self.scratches_manager.execute_all(image)
        if self.patches_manager:
            image = self.patches_manager.execute_all(image)

        processed_image = image

        # Ejecutar detección
        results = self.detector_manager.execute(processed_image)

        return results

    def _init_preprocessing_manager(self, defect_type):
        """Inicializa un manager de preprocesamiento para un tipo de defecto"""
        manager = PreprocessingManager()

        # Obtener métodos configurados
        methods_config = self.config.get(f"{defect_type}_preprocessing", [])

        if methods_config:
            # Usar métodos configurados en el JSON
            for method_config in methods_config:
                method_name = method_config.get("name")
                method_params = method_config.get("params", {})
                try:
                    method = Tools.create_instance(method_name, method_params, "metal.preprocessing")
                    manager.add_method(method)
                    self.logger.info(f"Método {method_name} añadido para {defect_type}")
                except Exception as e:
                    self.logger.error(f"Error añadiendo método {method_name}: {e}")
        else:
            # Usar valores predeterminados
            self.logger.info(f"Usando métodos predeterminados para {defect_type}")
            if defect_type == "scratches":
                manager.add_method(CLAHEMethod(clip_limit=2.5, grid_size=(8, 8)))
                manager.add_method(BrightScratchMethod(contrast_enhance=1.5, threshold_factor=0.7))
                manager.add_method(MorphologyMethod(operation='close', kernel_size=(3, 9)))
            elif defect_type == "patches":
                manager.add_method(GaussianBlurMethod(sigma=1.5))
                manager.add_method(LocalContrastMethod(kernel_size=25, contrast_factor=25))
                manager.add_method(AdaptiveThresholdMethod(block_size=35, C=7))
                manager.add_method(MorphologyMethod(operation='close', kernel_size=7))
                manager.add_method(MorphologyMethod(operation='open', kernel_size=3))

        # Asignar manager a la instancia
        setattr(self, f"{defect_type}_manager", manager)

    def _init_detector(self, defect_type):
        """Inicializa un detector para un tipo de defecto"""
        detector_config = self.config.get(f"{defect_type}_detector", {})

        if detector_config:
            # Usar detector configurado en el JSON
            detector_name = detector_config.get("name")
            detector_params = detector_config.get("params", {})
            try:
                detector = Tools.create_instance(detector_name, detector_params, "metal.detection")
                self.detector_manager = DetectorManager(detector)
                self.logger.info(f"Detector {detector_name} configurado para {defect_type}")
            except Exception as e:
                self.logger.error(f"Error inicializando detector para {defect_type}: {e}")
                self.detector_manager = self._create_default_detector(defect_type)
        else:
            # Usar detector predeterminado
            self.logger.info(f"Usando detector predeterminado para {defect_type}")
            self.detector_manager = self._create_default_detector(defect_type)

        # Asignar detector a la instancia
        setattr(self, f"{defect_type}_detector_manager", self.detector_manager)

    def _create_default_detector(self, defect_type):
        """Crea un detector predeterminado para un tipo de defecto"""
        if defect_type == "scratches":
            return DetectorManager(ScratchDetectionMethod(min_length=30, max_width=20, max_results=5))
        elif defect_type == "patches":
            return DetectorManager(EnhancedConnectedComponentsDetectionMethod(
                area_min=200, area_max=20000, max_results=5,
                border_threshold=15, aspect_ratio_limit=5
            ))