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
### Ejemplo de Archivo de Configuración

A continuación, se muestra un ejemplo de archivo de configuración en formato JSON:

```bash
{
"preprocessing_methods": [
"UmbralizeMethod",
"CannyMethod"
],
"detector_methods": "ContrastMethod"
}
```

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
