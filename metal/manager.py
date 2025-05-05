from metal.preprocessing import *
from metal.detection import *
from metal.tools import Tools

class MainManager:
    def __init__(self, config_path, image_path):
        self.config = Tools.parse_config(config_path)
        self.image_path = image_path
        self.scratches_manager = None
        self.patches_manager = None
        self.detector_manager = None

    def start(self):
        # Leer imagen
        image = Tools.read_image(self.image_path)

        # Configurar preprocesadores
        defect_type = self.config.get("defect_type", "auto")

        # Siempre se crean ambos managers



        if defect_type in ("scratches", "auto"):
            self.scratches_manager = PreprocessingManager()
            self.scratches_manager.add_method(CLAHEMethod(clip_limit=2.5, grid_size=(8, 8)))
            self.scratches_manager.add_method(BrightScratchMethod(contrast_enhance=1.5, threshold_factor=0.7))
            self.scratches_manager.add_method(MorphologyMethod(operation='close', kernel_size=(3, 9)))

        if defect_type in ("patches", "auto"):
            self.patches_manager = PreprocessingManager()
            self.patches_manager.add_method(GaussianBlurMethod(sigma=1.5))
            self.patches_manager.add_method(LocalContrastMethod(kernel_size=25, contrast_factor=25))
            self.patches_manager.add_method(AdaptiveThresholdMethod(block_size=35, C=7))
            self.patches_manager.add_method(MorphologyMethod(operation='close', kernel_size=7))
            self.patches_manager.add_method(MorphologyMethod(operation='open', kernel_size=3))

        # Configurar detector
        self.detector_manager = DetectorManager(EnhancedConnectedComponentsDetectionMethod(
            area_min=200,  # Área mínima mayor para evitar detecciones de ruido
            area_max=20000,  # Área máxima ajustada para defectos grandes
            max_results=5,  # Limitar a 5 detecciones como máximo
            border_threshold=15,  # Ignorar detecciones que estén a menos de 15 píxeles del borde
            aspect_ratio_limit=5  # Limitar la relación de aspecto para evitar detecciones muy alargadas
        ))

        if defect_type == "scratches":
            self.detector_manager = DetectorManager(ScratchDetectionMethod(min_length=30, max_width=20, max_results=5))
        elif defect_type == "patches":
            self.detector_manager = DetectorManager(EnhancedConnectedComponentsDetectionMethod(
                area_min=200,  # Área mínima mayor para evitar detecciones de ruido
                area_max=20000,  # Área máxima ajustada para defectos grandes
                max_results=5,  # Limitar a 5 detecciones como máximo
                border_threshold=15,  # Ignorar detecciones que estén a menos de 15 píxeles del borde
                aspect_ratio_limit=5  # Limitar la relación de aspecto para evitar detecciones muy alargadas
            ))
        else:  # auto
            self.detector_manager = DetectorManager(MultiDefectDetectionMethod(
                scratch_detector=ScratchDetectionMethod(min_length=30, max_width=20, max_results=5),
                patch_detector=EnhancedConnectedComponentsDetectionMethod(
                    area_min=200, area_max=20000, max_results=5, border_threshold=15, aspect_ratio_limit=5),
                combine_results=True
            ))

        # Ejecutar preprocesadores
        if self.scratches_manager:
            image = self.scratches_manager.execute_all(image)
        if self.patches_manager:
            image = self.patches_manager.execute_all(image)

        processed_image = image

        # Ejecutar detección
        results = self.detector_manager.execute(processed_image)

        # Mostrar resultados (puedes personalizar esto)
        for result in results:
            print(f"Detección: x={result.px}, y={result.py}, w={result.width}, h={result.height}")

        return results
