import cv2
import numpy as np
from abc import ABC, abstractmethod

class PreprocessingMethod(ABC):
    @abstractmethod
    def process(self, image):
        """Método abstracto que debe implementar cada método de preprocesado"""
        pass

class GaussianBlurMethod(PreprocessingMethod):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def process(self, image):
        return cv2.GaussianBlur(image, (0, 0), self.sigma)

class MedianBlurMethod(PreprocessingMethod):
    def __init__(self, ksize=3):
        self.ksize = ksize if self.ksize % 2 == 1 else self.ksize + 1  # ksize debe ser impar

    def process(self, image):
        return cv2.medianBlur(image, self.ksize)

class SobelGradientMethod(PreprocessingMethod):
    def __init__(self):
        pass

    def process(self, image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        grad = np.uint8(np.clip(grad, 0, 255))
        return grad

class ThresholdMethod(PreprocessingMethod):
    def __init__(self, factor=0.2):
        self.factor = factor

    def process(self, image):
        thresh = np.max(image) * self.factor
        return (image > thresh).astype(np.uint8) * 255


class AdaptiveThresholdMethod(PreprocessingMethod):
    def __init__(self, block_size=35, C=5, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.C = C
        self.adaptive_method = adaptive_method

    def process(self, image):

        # Asegurar que la imagen sea de tipo uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Asegurar que sea de un solo canal
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Suavizado para reducir ruido antes de umbralizar
        image = cv2.medianBlur(image, 3)

        return cv2.adaptiveThreshold(
            image, 255, self.adaptive_method,
            cv2.THRESH_BINARY_INV, self.block_size, self.C
        )


class MorphologyMethod(PreprocessingMethod):
    def __init__(self, operation='close', kernel_size=3, kernel_type=cv2.MORPH_RECT):
        self.operation = operation

        # Verificar si kernel_size ya es una tupla
        if isinstance(kernel_size, tuple):
            # Usar directamente como tamaño del kernel
            self.kernel = cv2.getStructuringElement(kernel_type, kernel_size)
        else:
            # Es un valor único, crear un kernel cuadrado
            self.kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))

    def process(self, image):

        # Asegurar que la imagen es binaria
        if image.dtype != np.uint8:
            image = (image > 0).astype(np.uint8) * 255

        if self.operation == 'erode':
            return cv2.erode(image, self.kernel)
        elif self.operation == 'dilate':
            return cv2.dilate(image, self.kernel)
        elif self.operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        elif self.operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)
        else:
            return image


class LocalContrastMethod(PreprocessingMethod):
    def __init__(self, kernel_size=25, contrast_factor=20, offset=128):
        self.kernel_size = kernel_size
        self.contrast_factor = contrast_factor
        self.offset = offset

    def process(self, image):
        # Convertir a float32 para cálculos
        image_float = image.astype(np.float32)

        # Aplicar un suavizado gaussiano para reducir ruido antes del contraste local
        image_float = cv2.GaussianBlur(image_float, (3, 3), 0)

        # Calcular media y desviación estándar locales
        mean_local = cv2.boxFilter(image_float, -1, (self.kernel_size, self.kernel_size), normalize=True)
        squared = image_float * image_float
        mean_squared = cv2.boxFilter(squared, -1, (self.kernel_size, self.kernel_size), normalize=True)
        std_local = np.sqrt(np.maximum(mean_squared - mean_local * mean_local, 0))

        # Realce de contraste con parámetros ajustables
        result = ((image_float - mean_local) / (std_local + 1e-5)) * self.contrast_factor + self.offset

        # Asegurar que el resultado esté en el rango 0-255 y sea uint8
        return np.uint8(np.clip(result, 0, 255))


class EnhancedPatchMethod(PreprocessingMethod):
    def process(self, image):

        # Convertir a escala de grises si es necesario
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Suavizado inicial para reducir ruido
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # 2. Realce de contraste local ajustado para manchas
        img_float = blurred.astype(np.float32)
        mean_local = cv2.boxFilter(img_float, -1, (25, 25), normalize=True)
        squared = img_float * img_float
        mean_squared = cv2.boxFilter(squared, -1, (25, 25), normalize=True)
        std_local = np.sqrt(np.maximum(mean_squared - mean_local * mean_local, 0))

        contrasted = ((img_float - mean_local) / (std_local + 1e-5)) * 25 + 128
        contrasted = np.uint8(np.clip(contrasted, 0, 255))

        # 3. Umbralización adaptativa con parámetros optimizados para manchas
        binary = cv2.adaptiveThreshold(
            contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 7
        )

        # 4. Operaciones morfológicas para conectar regiones fragmentadas
        # Aplicar cierre morfológico para conectar fragmentos
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # Remover ruido pequeño con apertura
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        return final


class CLAHEMethod(PreprocessingMethod):
    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def process(self, image):

        # Asegurar que la imagen es de tipo uint8 y en escala de grises
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        return clahe.apply(image)


class DirectionalFilterMethod(PreprocessingMethod):
    def __init__(self, orientations=[0, 45, 90, 135], kernel_size=15):
        self.orientations = orientations
        self.kernel_size = kernel_size

    def process(self, image):

        # Asegurar formato correcto
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convertir a float para operaciones
        image_float = image.astype(np.float32)

        # Resultados de filtros en diferentes orientaciones
        results = []

        for angle in self.orientations:
            # Crear kernel direccional
            size = self.kernel_size
            kernel = np.zeros((size, size), dtype=np.float32)

            # Llenar kernel según orientación
            if angle == 0:  # Horizontal
                kernel[size // 2, :] = 1
            elif angle == 90:  # Vertical
                kernel[:, size // 2] = 1
            elif angle == 45:  # Diagonal 45°
                for i in range(size):
                    if 0 <= i < size and 0 <= i < size:
                        kernel[i, i] = 1
            elif angle == 135:  # Diagonal 135°
                for i in range(size):
                    if 0 <= i < size and 0 <= size - i - 1 < size:
                        kernel[i, size - i - 1] = 1

            # Normalizar kernel
            kernel = kernel / np.sum(kernel)

            # Aplicar filtro
            filtered = cv2.filter2D(image_float, -1, kernel)

            # Restar imagen original para resaltar diferencias
            enhanced = cv2.absdiff(filtered, image_float)

            results.append(enhanced)

        # Combinar resultados (máximo en cada píxel)
        max_positions = np.zeros_like(image, dtype=np.uint8)
        for result in results:
            # Detectar picos locales (posibles centros de rayones)
            local_max = cv2.dilate(result, np.ones((3, 3), np.uint8))
            local_max = (result == local_max) & (result > np.mean(result) + np.std(result))
            max_positions = np.maximum(max_positions, local_max.astype(np.uint8) * 255)

        # Combinar información de posición con la detección final
        combined = np.zeros_like(results[0])
        for result in results:
            combined = np.maximum(combined, result)

        # Asegurar que los centros detectados se preserven en la imagen final
        final = np.maximum(combined, max_positions)

        return final

class BrightScratchMethod(PreprocessingMethod):
    def __init__(self, contrast_enhance=1.5, threshold_factor=0.7):
        self.contrast_enhance = contrast_enhance
        self.threshold_factor = threshold_factor

    def process(self, image):

        # Asegurar escala de grises
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Mejorar contraste para resaltar elementos brillantes
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # 2. Aplicar umbralización para destacar solo elementos brillantes
        # Calcular umbral adaptativo basado en histograma
        hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
        total_pixels = enhanced.shape[0] * enhanced.shape[1]

        # Encontrar umbral que separe el top 10-15% más brillante
        cumsum = 0
        for i in range(255, -1, -1):
            cumsum += hist[i][0]
            if cumsum / total_pixels > 0.15:  # Ajustar este valor según necesidades
                threshold = i
                break

        binary = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)[1]

        # 3. Aplicar operaciones morfológicas específicas para rayones
        # Kernel direccional vertical alargado (para rayones verticales)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        # Kernel direccional horizontal (para rayones horizontales)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))

        # Aplicar aperturas direccionales para eliminar ruido pequeño
        opened_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
        opened_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)

        # Combinar resultados
        combined = cv2.bitwise_or(opened_v, opened_h)

        # 4. Conectar fragmentos del mismo rayón
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

        return closed

class AdaptiveStatsThresholdMethod(PreprocessingMethod):
    def __init__(self, std_factor=1.5, offset=0):
        self.std_factor = std_factor
        self.offset = offset

    def process(self, image):

        # Asegurar formato correcto
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calcular estadísticas globales
        mean_val = np.mean(image)
        std_val = np.std(image)

        # Calcular umbral adaptativo
        threshold = mean_val + (self.std_factor * std_val) + self.offset

        # Aplicar umbralización
        binary = (image > threshold).astype(np.uint8) * 255

        return binary


class InvertMethod(PreprocessingMethod):
    def __init__(self):
        pass

    def process(self, image):
        return 255 - image

class NormalizeMethod(PreprocessingMethod):
    def __init__(self):
        pass

    def process(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            norm = 255.0 * (image - min_val) / (max_val - min_val)
        else:
            norm = image.copy()
        return np.uint8(norm)

class UmbralizeMethod(PreprocessingMethod):
    def process(self, image):
        _, processed_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        return processed_image

class CannyMethod(PreprocessingMethod):
    def process(self, image):
        processed_image = cv2.Canny(image, threshold1=100, threshold2=100 * 2)
        return processed_image

class PreprocessingManager:
    def __init__(self):
        self.methods = []

    def add_method(self, method: PreprocessingMethod):
        self.methods.append(method)

    def execute_all(self, image):
        for method in self.methods:
            image = method.process(image)
        return image
