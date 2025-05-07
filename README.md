# Software de Detección de Desperfectos en Placas Metálicas

Este proyecto implementa un sistema de visión artificial diseñado para la inspección automática de placas metálicas en entornos industriales. Su funcionalidad principal es procesar imágenes de las superficies de los metales para detectar desperfectos (arañazos, manchas, etc.) y generar las coordenadas de los “bounding boxes” correspondientes. El sistema está diseñado para ejecutarse en Linux y es compatible con plataformas embebidas como Raspberry Pi, pudiendo integrarse tanto en aplicaciones Java como en entornos Python.

## Características del Proyecto

- **Procesamiento de imágenes**
Corrige la iluminación y reduce el ruido de imágenes en escala de grises de 200x200 píxeles.
- **Detección de defectos**
Identifica hasta 5 defectos por imagen, devolviendo una lista de tuplas con las coordenadas y dimensiones del bounding box. En ausencia de defectos se devuelve la tupla (0,0,0,0).
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

**Método para instalación en nativo con Python**

- Clona el repositorio y accede a la carpeta:

```bash
git clone https://github.com/ArturoLara/AIVA_2024_METALGROUP.git
cd AIVA_2024_METALGROUP
```

- (Opcional) Crea y activa un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

- Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

> Consulta el archivo `requirements.txt` para la lista completa y actualizada de dependencias.

---

**Método Docker**

- Asegúrate de tener Docker instalado en tu sistema.
- Descarga y ejecuta la imagen oficial desde Docker Hub, montando el directorio actual para acceder a tus imágenes y archivos de configuración:

```bash
docker pull artzulm/aiva_2024_metalgroup:latest
```

> El contenedor Docker incluye todas las dependencias y el código necesario, facilitando la ejecución sin necesidad de configurar el entorno Python manualmente.

---

## Uso

El proyecto está diseñado de manera modular para facilitar su integración y adaptarse a diferentes necesidades. Actualmente, la ejecución principal se realiza mediante un script `main` que permite analizar imágenes y detectar imperfecciones utilizando parámetros configurables. A continuación, se detalla el uso:

### Ejecución del Script Principal

El programa se ejecuta desde la línea de comandos utilizando el siguiente formato:

```bash
python main.py --config  --image 
```

### Parámetros

- **`--config`**: Ruta al archivo de configuración en formato JSON.
- **`--image`**: Ruta a la imagen que se desea analizar. La imagen debe tener un tamaño adecuado para garantizar resultados óptimos.

### Ejemplo Básico

Suponiendo un archivo de configuración `config.json` y una imagen `imagen.png`. Se puede ejecutar de la siguiente manera:

```bash
python main.py --config config.json --image imagen.png
```

## Sistema de Configuración de Detección de Defectos

### Estructura del JSON
El archivo de configuración permite definir pipelines personalizados para cada tipo de defecto. La estructura básica es:

```json
{
    "defect_type": "auto",
    "scratches_preprocessing": [],
    "patches_preprocessing": [],
    "scratches_detector": {},
    "patches_detector": {}
}
```

#### Secciones Clave:
1. **`defect_type`**:  
   - `"scratches"`: Solo detecta arañazos  
   - `"patches"`: Solo detecta manchas  
   - `"auto"`: Detecta ambos tipos (valor por defecto)

2. **Preprocesado (`*_preprocessing`)**  
   Lista de métodos a aplicar en orden, cada uno con sus parámetros:
   ```json
   {
       "name": "NombreClaseMetodo",
       "params": {"param1": valor1, "param2": valor2}
   }
   ```

3. **Detección (`*_detector`)**  
   Configuración específica para cada tipo de detector:
   ```json
   {
       "name": "NombreClaseDetector",
       "params": {"param1": valor1}
   }
   ```

---

### Métodos Disponibles

| **Tipo**      | **Clase / Método**                        | **Parámetros**                                                                                      | **Descripción**                                               |
|---------------|-------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Preprocesado  | `GaussianBlurMethod`                      | `sigma` (float, por defecto 1.0)                                                                   | Suavizado Gaussiano                                           |
| Preprocesado  | `MedianBlurMethod`                        | `ksize` (int, impar, por defecto 3)                                                                | Suavizado Mediano                                             |
| Preprocesado  | `SobelGradientMethod`                     | *(sin parámetros)*                                                                                  | Gradiente Sobel (bordes)                                      |
| Preprocesado  | `ThresholdMethod`                         | `factor` (float, por defecto 0.2)                                                                  | Umbralización global                                          |
| Preprocesado  | `AdaptiveThresholdMethod`                 | `block_size` (int, impar, por defecto 35), `C` (int, por defecto 5)                                | Umbralización adaptativa                                      |
| Preprocesado  | `MorphologyMethod`                        | `operation` (str), `kernel_size` (int o tupla), `kernel_type` (por defecto MORPH_RECT)             | Operaciones morfológicas (open, close, erode, dilate)         |
| Preprocesado  | `LocalContrastMethod`                     | `kernel_size` (int, por defecto 25), `contrast_factor` (int, por defecto 20), `offset` (int, 128)  | Realce de contraste local                                     |
| Preprocesado  | `EnhancedPatchMethod`                     | *(sin parámetros)*                                                                                  | Pipeline especializado para manchas                           |
| Preprocesado  | `CLAHEMethod`                             | `clip_limit` (float, por defecto 2.0), `grid_size` (tupla, por defecto (8,8))                      | Equalización adaptativa de histograma                         |
| Preprocesado  | `DirectionalFilterMethod`                 | `orientations` (lista de int, por defecto[135]), `kernel_size` (int, por defecto 15)      | Filtrado direccional                                          |
| Preprocesado  | `BrightScratchMethod`                     | `contrast_enhance` (float, 1.5), `threshold_factor` (float, 0.7)                                   | Realce y umbral para rayones brillantes                       |
| Preprocesado  | `AdaptiveStatsThresholdMethod`            | `std_factor` (float, 1.5), `offset` (int, 0)                                                       | Umbralización estadística local                               |
| Preprocesado  | `InvertMethod`                            | *(sin parámetros)*                                                                                  | Inversión de intensidades                                     |
| Preprocesado  | `NormalizeMethod`                         | *(sin parámetros)*                                                                                  | Normalización de rango dinámico                               |
| Preprocesado  | `UmbralizeMethod`                         | *(sin parámetros)*                                                                                  | Umbralización fija a 200                                      |
| Preprocesado  | `CannyMethod`                             | *(sin parámetros)*                                                                                  | Detección de bordes Canny                                     |
| Detección     | `ContrastMethod`                          | *(sin parámetros)*                                                                                  | Detección por contornos                                       |
| Detección     | `ConnectedComponentsDetectionMethod`      | `area_min` (int, 50), `area_max` (int, 5000), `max_results` (int, 5)                               | Componentes conectados básico                                 |
| Detección     | `EnhancedConnectedComponentsDetectionMethod` | `area_min` (200), `area_max` (20000), `max_results` (5), `border_threshold` (10), `aspect_ratio_limit` (8) | Componentes conectados avanzado                               |
| Detección     | `ScratchDetectionMethod`                  | `min_length` (30), `max_width` (20), `max_results` (5)                                             | Detección de rayones                                          |
| Detección     | `MultiDefectDetectionMethod`              | `scratch_detector`, `patch_detector`, `combine_results` (bool, True)                               | Combinación de detectores especializados                      |

---

### Ejemplos Prácticos

#### 1. Configuración para arañazos con realce de contraste
```json
{
    "defect_type": "scratches",
    "scratches_preprocessing": [
        {
            "name": "CLAHEMethod",
            "params": {"clip_limit": 3.0}
        },
        {
            "name": "MorphologyMethod",
            "params": {"operation": "close", "kernel_size": [3,9]}
        }
    ]
}
```

#### 2. Detección automática con parámetros personalizados
```json
{
    "defect_type": "auto",
    "patches_preprocessing": [
        {
            "name": "MedianBlurMethod",
            "params": {"ksize": 5}
        },
        {
            "name": "AdaptiveThresholdMethod",
            "params": {"block_size": 25}
        }
    ],
    "scratches_detector": {
        "name": "ScratchDetectionMethod",
        "params": {"min_length": 40}
    }
}
```

---

**Uso con Python**

El sistema se utiliza mediante la ejecución directa del script principal, pasando la ruta al archivo de configuración y la imagen a analizar. Ejemplo básico:

```python
from metal.manager import MainManager

config_path = "ruta/a/config.json"
image_path = "ruta/a/tu_imagen.jpg"

manager = MainManager(config_path=config_path, image_path=image_path)
detections = manager.start()
print("Defectos detectados:", detections)
```

Para ejecutar desde línea de comandos:

```bash
python main.py --config ruta/a/config.json --image ruta/a/tu_imagen.jpg
```

- El resultado será una lista de objetos detectados. Puedes usar la función `dibujar_rectangulos_y_guardar` para visualizar los defectos sobre la imagen original y guardar el resultado:

```python
from main import dibujar_rectangulos_y_guardar

dibujar_rectangulos_y_guardar(image_path, detections, "output_patches.jpg")
```

---

**Uso con Docker**

- Una vez tengas tus archivos en el directorio actual (imagen y configuración), ejecuta:

```bash
docker run -e CONFIG=/App/config.json -e IMAGE=/App/tu_imagen.jpg -v $(pwd):/App artzulm/aiva_2024_metalgroup:latest
```
- Sustituye `tu_imagen.jpg` y `config.json` por los nombres de tus archivos dentro del directorio actual.
- El contenedor procesará la imagen indicada y generará la salida en el mismo mormato que el método Python para su versión asociada, pero en un entorno aislado y controlado.

> Este método encapsula todo el flujo de análisis y detección en un entorno controlado, ideal para despliegues rápidos y reproducibles.

---

Ambos métodos (Python y Docker) permiten analizar imágenes para detectar defectos, generando como salida una lista de objetos detectados y, opcionalmente, una imagen con los defectos resaltados. Docker facilita la integración y despliegue sin preocuparse por dependencias o configuraciones adicionales.

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
