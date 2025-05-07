import numpy as np
from abc import ABC, abstractmethod
import cv2
from scipy import ndimage

class DetectionResult:
    def __init__(self, px, py, width, height):
        self.px = px
        self.py = py
        self.width = width
        self.height = height

    def __iter__(self):
        return iter((self.px, self.py, self.width, self.height))

class DetectionMethod(ABC):

    @abstractmethod
    def detect(self, image):
        """Método que debe implementar el algoritmo de detección"""
        pass

class ContrastMethod(DetectionMethod):
    def detect(self, image):
        # Encontrar los contornos en la imagen de bordes
        contornos, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lista para almacenar las zonas detectadas como objetos DetectionResult
        zonas_detectadas = []

        for contorno in contornos:
            # Calcular el rectángulo delimitador para cada contorno
            x, y, w, h = cv2.boundingRect(contorno)
            zonas_detectadas.append(DetectionResult(x, y, w, h))

        # Ordenar las zonas detectadas por área en orden descendente
        zonas_detectadas.sort(key=lambda zona: zona.width * zona.height, reverse=True)

        # Filtrar zonas que se superponen
        zonas_filtradas = []
        for i, zona in enumerate(zonas_detectadas):
            x1, y1, w1, h1 = zona.px, zona.py, zona.width, zona.height
            area1 = w1 * h1

            superpuesto = False
            for j in range(i):
                x2, y2, w2, h2 = zonas_detectadas[j].px, zonas_detectadas[j].py, zonas_detectadas[j].width, \
                zonas_detectadas[j].height
                area2 = w2 * h2

                # Calcular intersección entre las dos áreas
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    if inter_area / area1 > 0.7 or inter_area / area2 > 0.7:
                        superpuesto = True
                        break

            if not superpuesto:
                zonas_filtradas.append(zona)

            if len(zonas_filtradas) == 5:
                break

        return DetectionResult(0, 0, 0, 0) if len(zonas_filtradas) == 0 else zonas_filtradas

class ConnectedComponentsDetectionMethod(DetectionMethod):
    def __init__(self, area_min=50, area_max=5000, max_results=5):
        self.area_min = area_min
        self.area_max = area_max
        self.max_results = max_results

    def detect(self, image):
        # Asegurar que la imagen es binaria
        imagen_binaria = image > 0

        # Etiquetar componentes conectados
        etiquetada, num_componentes = ndimage.label(imagen_binaria)

        # Calcular propiedades de los objetos
        objetos = ndimage.find_objects(etiquetada)

        # Filtrar objetos por área y crear DetectionResult
        zonas_detectadas = []
        for i, obj in enumerate(objetos):
            if obj is not None:
                area = np.sum(etiquetada[obj] == i + 1)
                if self.area_min <= area <= self.area_max:
                    y_min, x_min = obj[0].start, obj[1].start
                    y_max, x_max = obj[0].stop, obj[1].stop
                    w = x_max - x_min
                    h = y_max - y_min
                    zonas_detectadas.append(DetectionResult(x_min, y_min, w, h))

        # Ordenar por área descendente y limitar a max_results
        zonas_detectadas.sort(key=lambda zona: zona.width * zona.height, reverse=True)
        zonas_filtradas = zonas_detectadas[:self.max_results]

        # Si no hay detecciones, devolver una lista con un DetectionResult vacío
        if not zonas_filtradas:
            return [DetectionResult(0, 0, 0, 0)]
        return zonas_filtradas


class EnhancedConnectedComponentsDetectionMethod(DetectionMethod):
    def __init__(self, area_min=200, area_max=20000, max_results=5, border_threshold=10, aspect_ratio_limit=8):
        self.area_min = area_min
        self.area_max = area_max
        self.max_results = max_results
        self.border_threshold = border_threshold
        self.aspect_ratio_limit = aspect_ratio_limit

    def detect(self, image):
        # Asegurar que la imagen es binaria
        imagen_binaria = (image > 0).astype(np.uint8)

        # Preprocesar la imagen con operación de cierre para unir regiones cercanas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        imagen_cerrada = cv2.morphologyEx(imagen_binaria * 255, cv2.MORPH_CLOSE, kernel)

        # Convertir explícitamente a uint8
        imagen_cerrada_uint8 = (imagen_cerrada > 0).astype(np.uint8)

        # Usar connectedComponentsWithStats directamente
        retval, etiquetada, stats, centroids = cv2.connectedComponentsWithStats(
            imagen_cerrada_uint8, 8, cv2.CV_32S
        )

        # Asegurar que num_componentes sea un entero escalar
        num_componentes = int(retval)

        # Filtrar componentes por área y posición
        zonas_detectadas = []
        for i in range(1, num_componentes):  # Ahora num_componentes es un entero
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Calcular relación de aspecto
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

            # Verificar si es un componente en el borde
            is_border = (x < self.border_threshold or
                         y < self.border_threshold or
                         x + w > image.shape[1] - self.border_threshold or
                         y + h > image.shape[0] - self.border_threshold)

            # Filtrar por área, relación de aspecto y posición en el borde
            if (self.area_min <= area <= self.area_max and
                    aspect_ratio <= self.aspect_ratio_limit and
                    (not is_border or area > self.area_min * 3)):  # Permitir componentes de borde solo si son grandes
                zonas_detectadas.append(DetectionResult(x, y, w, h))

        # Aplicar Non-Maximum Suppression para eliminar detecciones redundantes
        if len(zonas_detectadas) > 1:
            zonas_detectadas = self._non_max_suppression(zonas_detectadas, 0.5)

        # Ordenar por área descendente
        zonas_detectadas.sort(key=lambda zona: zona.width * zona.height, reverse=True)
        zonas_detectadas = zonas_detectadas[:self.max_results]

        # Si no hay detecciones válidas, devolver una lista con un DetectionResult vacío
        if not zonas_detectadas:
            return [DetectionResult(0, 0, 0, 0)]
        return zonas_detectadas

    def _non_max_suppression(self, boxes, overlap_thresh):
        """Aplica non-maximum suppression para eliminar detecciones redundantes"""
        if len(boxes) == 0:
            return []

        # Convertir a formato de coordenadas para NMS
        rects = [(box.px, box.py, box.px + box.width, box.py + box.height) for box in boxes]

        # Calcular áreas
        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in rects]

        # Ordenar por área descendente
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

        keep = []
        while indices:
            i = indices[0]
            keep.append(i)

            indices.pop(0)

            if not indices:
                break

            x1i, y1i, x2i, y2i = rects[i]
            areai = areas[i]

            # Calcular solapamiento con todas las restantes
            indices_to_remove = []
            for j_idx, j in enumerate(indices):
                x1j, y1j, x2j, y2j = rects[j]

                # Calcular intersección
                xx1 = max(x1i, x1j)
                yy1 = max(y1i, y1j)
                xx2 = min(x2i, x2j)
                yy2 = min(y2i, y2j)

                # Calcular área de intersección
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter_area = w * h

                # Calcular IoU
                iou = inter_area / (areai + areas[j] - inter_area)

                if iou > overlap_thresh:
                    indices_to_remove.append(j_idx)

            # Eliminar índices en orden inverso para no afectar los índices anteriores
            for idx in sorted(indices_to_remove, reverse=True):
                indices.pop(idx)

        return [boxes[i] for i in keep]


class ScratchDetectionMethod(DetectionMethod):
    def __init__(self, min_length=30, max_width=20, max_results=5):
        self.min_length = min_length
        self.max_width = max_width
        self.max_results = max_results

    def detect(self, image):
        # Asegurar que la imagen sea binaria
        imagen_binaria = (image > 0).astype(np.uint8)

        # Usar connectedComponentsWithStats
        retval, etiquetada, stats, centroids = cv2.connectedComponentsWithStats(
            imagen_binaria, 8, cv2.CV_32S
        )
        num_componentes = int(retval)

        # Filtrar componentes para identificar líneas (rayones)
        zonas_detectadas = []
        for i in range(1, num_componentes):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Calcular relación de aspecto y cobertura
            aspect_ratio = max(h, w) / (min(h, w) + 1e-5)
            area_ratio = area / (w * h)

            # Los rayones típicamente son líneas alargadas (relación de aspecto alta)
            is_line_like = aspect_ratio > 3

            # Para rayones, queremos estructuras largas pero no muy anchas
            length = max(h, w)
            width = min(h, w)

            if (length >= self.min_length and width <= self.max_width and is_line_like):
                # Si el rayón es horizontal, asegurar que x,y,w,h representen correctamente su orientación
                if w > h:  # Horizontal
                    zonas_detectadas.append(DetectionResult(x, y, w, h))
                else:  # Vertical u orientación arbitraria
                    zonas_detectadas.append(DetectionResult(x, y, w, h))

        # Ordenar por tamaño (priorizando los rayones más largos)
        zonas_detectadas.sort(key=lambda zona: max(zona.width, zona.height), reverse=True)
        zonas_detectadas = zonas_detectadas[:self.max_results]

        # Si no hay detecciones válidas, devolver una lista con un DetectionResult vacío
        if not zonas_detectadas:
            return [DetectionResult(0, 0, 0, 0)]
        return zonas_detectadas


class MultiDefectDetectionMethod(DetectionMethod):
    def __init__(self, scratch_detector, patch_detector, combine_results=True):
        self.scratch_detector = scratch_detector
        self.patch_detector = patch_detector
        self.combine_results = combine_results

    def detect(self, image):


        # Crear dos copias de la imagen para detección independiente
        scratch_image = image.copy()
        patch_image = image.copy()

        # Aplicar detectores especializados
        scratch_results = self.scratch_detector.detect(scratch_image)
        patch_results = self.patch_detector.detect(patch_image)

        # Eliminar detecciones vacías (0,0,0,0)
        valid_scratch_results = [r for r in scratch_results if r.width > 0 and r.height > 0]
        valid_patch_results = [r for r in patch_results if r.width > 0 and r.height > 0]

        if not self.combine_results:
            # Devolver ambos resultados separados
            return valid_scratch_results, valid_patch_results

        # Combinar resultados
        combined_results = valid_scratch_results + valid_patch_results

        # Ordenar por área
        combined_results.sort(key=lambda r: r.width * r.height, reverse=True)

        # Aplicar non-maximum suppression para eliminar solapamientos
        filtered_results = self._non_max_suppression(combined_results, 0.5)

        # Limitar a máximo 5 resultados
        final_results = filtered_results[:5]

        # Si no hay detecciones, devolver una vacía
        if not final_results:
            return [DetectionResult(0, 0, 0, 0)]

        return final_results

    def _non_max_suppression(self, boxes, overlap_thresh):
        """Elimina detecciones redundantes mediante NMS"""
        if len(boxes) == 0:
            return []

        # Convertir a formato de coordenadas para NMS
        rects = [(box.px, box.py, box.px + box.width, box.py + box.height) for box in boxes]

        # Calcular áreas
        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in rects]

        # Ordenar por área descendente
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

        keep = []
        while indices:
            i = indices[0]
            keep.append(i)

            indices.pop(0)

            if not indices:
                break

            x1i, y1i, x2i, y2i = rects[i]
            areai = areas[i]

            # Calcular solapamiento con las restantes
            indices_to_remove = []
            for j_idx, j in enumerate(indices):
                x1j, y1j, x2j, y2j = rects[j]

                # Calcular intersección
                xx1 = max(x1i, x1j)
                yy1 = max(y1i, y1j)
                xx2 = min(x2i, x2j)
                yy2 = min(y2i, y2j)

                # Calcular área de intersección
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter_area = w * h

                # Calcular IoU
                iou = inter_area / (areai + areas[j] - inter_area)

                if iou > overlap_thresh:
                    indices_to_remove.append(j_idx)

            # Eliminar índices en orden inverso
            for idx in sorted(indices_to_remove, reverse=True):
                indices.pop(idx)

        return [boxes[i] for i in keep]


class DetectorManager:
    def __init__(self, method: DetectionMethod):
        self.method = method

    def execute(self, image):
        return self.method.detect(image)
