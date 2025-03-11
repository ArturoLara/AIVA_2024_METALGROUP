import numpy as np

def detect_defects(image):
    """
    Función mockup para detectar defectos en una imagen procesada.

    Si la imagen es la dummy (todos sus píxeles en cero),
    se asume que no hay defectos y se retorna [(0,0,0,0)].

    En otro caso se simula la detección de un defecto con coordenadas dummy.
    """
    # Si la imagen es completamente negra, se interpreta como sin defectos.
    if np.sum(image) == 0:
        return [(0, 0, 0, 0)]
    else:
        # Se simula la detección de un defecto con coordenadas fijas.
        return [(10, 10, 50, 50)]
