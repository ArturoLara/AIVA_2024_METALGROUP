# Software de Detección de Desperfectos en Placas Metálicas

Este proyecto implementa un sistema de visión artificial diseñado para la inspección automática de placas metálicas en entornos industriales. Su funcionalidad principal es procesar imágenes de las superficies de los metales para detectar desperfectos (arañazos, manchas, etc.) y generar las coordenadas de los “bounding boxes” correspondientes. El sistema está diseñado para ejecutarse en Linux y es compatible con plataformas embebidas como Raspberry Pi, pudiendo integrarse tanto en aplicaciones Java como en entornos Python.

## Características del Proyecto

- **Procesamiento de imágenes**
Corrige la iluminación y reduce el ruido de imágenes en escala de grises de 200x200 píxeles.
- **Detección de defectos**

Identifica hasta 5 defectos por imagen, devolviendo una lista de tuplas con las coordenadas y dimensiones del bounding box. En ausencia de defectos se devuelve la tupla $$
(0,0,0,0)
$$.
- **Rendimiento y precisión**
    - Tiempo máximo de procesamiento: 200 ms por imagen.
    - Precisión mínima del 90 % medida en F1-score, usando un umbral de 80 % en la IoU (Intersection over Union).
- **Integración multiplataforma**
La solución permite la invocación de sus funcionalidades desde Java y Python, facilitando su incorporación en diversos sistemas de inspección industrial.


## Requisitos del Sistema

- **Sistema Operativo:** Linux (compatible con distribuciones como Ubuntu, Debian, etc.)
- **Hardware:**
    - Procesador con capacidad suficiente para procesamiento de imágenes en tiempo real.
    - Compatibilidad con plataformas embebidas (por ejemplo, Raspberry Pi).
- **Lenguaje de programación:** Python (se recomienda Python 3.8 o superior)
- **Bibliotecas y frameworks:**
    - [OpenCV](https://opencv.org/)
    - NumPy y SciPy
    - *(Opcional)* TensorFlow o PyTorch en caso de incorporar modelos de aprendizaje automático.


## Instalación y Configuración

1. **Clonar el repositorio:**

```bash
git clone https://github.com/tu_usuario/defect-detection.git
cd defect-detection
```

2. **Crear y activar un entorno virtual (opcional):**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar las dependencias:**

```bash
pip install -r requirements.txt
```


> Se recomienda revisar el archivo `requirements.txt` para obtener la lista actualizada de dependencias.

## Uso

El proyecto se organiza de manera modular para facilitar su integración. A modo de ejemplo:

- **Procesado de imágenes:**
Ejecuta el módulo de preprocesamiento proporcionando la ruta de la imagen (200x200 píxeles). El sistema devolverá una imagen optimizada para el análisis.
- **Detección de defectos:**
Invoca la función principal de detección pasando la imagen preprocesada. La salida será una lista de tuplas con la información de cada defecto detectado o $$
[(0,0,0,0)]
$$ en caso de no haber defectos.

Ejemplo básico en Python:

```python
from detection_module import detectar_defectos

ruta_imagen = "ruta/a/tu/imagen.png"
resultado = detectar_defectos(ruta_imagen)
print("Defectos detectados:", resultado)
```

> La función está diseñada para integrarse fácilmente tanto en proyectos Python como en aplicaciones Java mediante interfaces externas.

## Pruebas y Validación

- **Pruebas unitarias:**
Se ha configurado un conjunto de pruebas unitarias que aseguran una cobertura mínima del 80 % del código fuente. Puedes ejecutar las pruebas con:

```bash
python -m unittest discover 
```

- **Validación de rendimiento:**
Se realizan evaluaciones de tiempo de procesamiento y precisión (F1-score) usando conjuntos de imágenes etiquetadas. Verifica que el 95 % de las imágenes se procesen en menos de 200 ms.


## Estructura del Proyecto

| Carpeta/Archivo    | Descripción |
|:-------------------| :-- |
| `/metal`              | Código fuente del sistema de detección |
| `/tests`           | Pruebas unitarias e integradas |
| `requirements.txt` | Lista de dependencias del proyecto |
| `README.md`        | Documentación e instrucciones generales del proyecto |

## Contribuciones

No están permitidas las contribuciones de personal externo a la organización.

Se tendrán en cuenta reportes de errores y sugerencias de mejoras que se indiquen en el repositorio por medio de issues.

## Licencia

Este proyecto es propiedad de IberVision, Inc. Spain. La distribución y el uso del código se regirán por la licencia definida en el archivo `LICENSE`.

## Contacto

Para consultas, sugerencias o reportar incidencias, por favor contacta a alguno de los desarrolladores:

- Adrián Cobo Merino
- Arturo Lara Martínez