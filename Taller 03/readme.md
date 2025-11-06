# Taller 3 – Sistema de Recuperación Texto-Imagen con CLIP y Caltech-256

Este proyecto implementa un **sistema de recuperación de imágenes a partir de texto** utilizando el modelo **CLIP (Contrastive Language–Image Pre-training)** de OpenAI, aplicado sobre el dataset **Caltech-256**.  
El objetivo es que, dada una consulta textual (ej. *"guitarra eléctrica"*, *"guantes de boxeo"*), el sistema recupere las imágenes más similares semánticamente dentro del conjunto de datos.

---

## Autores
- **María Sofía Uribe**  
- **Javier Daza Olivella**

---

## Contenido del Notebook


### Configuración Inicial
- Se define el modelo CLIP a utilizar y el dispositivo (`CPU`/`GPU`).
- Se establecen parámetros como tamaño de batch y cantidad de imágenes a muestrear.
- Se configura la visualización del top-k de resultados (top 12).

### Carga del Dataset Caltech-256
- Se descarga el dataset desde el repositorio oficial.
- Se organizan las rutas de las imágenes, nombres de clases y etiquetas.
- Se valida la estructura de directorios y conteo total de imágenes: **~29.780 imágenes en 257 clases**.

### Muestreo y Verificación del Dataset
- Se selecciona una muestra balanceada de ~5.900 imágenes (256 clases, excluyendo "clutter").
- Se comprueba que los paths sean válidos y se analiza la distribución de clases.

### Generación de Embeddings con CLIP
- Se generan los embeddings de las imágenes usando la red visual de CLIP.
- Se codifican prompts textuales para establecer un espacio latente común entre texto e imagen.

### Construcción del Índice de Búsqueda
- Se crea un **índice vectorial con FAISS** para realizar búsquedas rápidas por similitud coseno.
- Se almacena:  
  ✔ embeddings de imágenes  
  ✔ etiquetas de clase  
  ✔ rutas de archivo

### Recuperación Texto → Imagen
Se implementa el flujo completo:
1. El usuario ingresa una consulta textual.
2. El texto se codifica con CLIP.
3. Se buscan los *k* vectores más similares en el índice FAISS.
4. Se muestran las imágenes recuperadas.

Ejemplos incluidos:
- `"guantes"` → guantes de boxeo, béisbol  
- `"guitarras"` → guitarras acústicas y eléctricas  
- `"animales"` → gorilas, jirafas, ballenas, etc.  
- `"deportes"` → pelotas, raquetas, ropa deportiva

### Evaluación del Sistema
- No se utilizan captions; se evalúa usando nombres de clase como texto.
- Se calcula precisión top-k verificando si alguna imagen del mismo label aparece en el resultado.
- Resultados muestran que CLIP alinea correctamente conceptos visuales con descripciones textuales.

### Exportación para Uso en Otros Entornos
Se muestra cómo guardar:
- los embeddings generados,  
- el índice FAISS,  
- y los metadatos del dataset  
para reutilizarlos posteriormente sin recalcular todo.

---

## Requisitos

| Requisito | Versión / Recomendación |
|-----------|---------------------------|
| Python    | 3.8+                     |
| PyTorch   | GPU recomendado          |
| Librerías | clip, faiss, numpy, PIL, matplotlib |

---

## Cómo Ejecutarlo

```bash
# Clonar repositorio
git clone <url-del-repo>
cd taller3

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook
jupyter notebook taller3.ipynb