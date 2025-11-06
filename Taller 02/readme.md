
# Taller 2 - Clasificación de Imágenes con Features Pre-entrenadas y AutoGluon

Este notebook implementa un sistema de clasificación de imágenes usando características extraídas de modelos pre-entrenados (ResNet50 y MobileNetV2) y clasificadores optimizados con AutoGluon.

## Autores
* María Sofía Uribe
* Javier Daza Olivella

## Dataset
Se utiliza el dataset LabelMe-12-50k que contiene imágenes clasificadas en 13 categorías:
- person
- car
- building
- window
- tree 
- sign
- door
- bookshelf
- chair
- table
- keyboard
- head
- clutter (regiones aleatorias sin objeto centrado)

## Metodología

### 1. Extracción de Características
- Se utilizan dos modelos pre-entrenados como extractores de features:
  - ResNet50
  - MobileNetV2
- Las imágenes se redimensionan a 224x224 píxeles
- Se aplica normalización usando la media y desviación estándar de ImageNet
- Las características extraídas se reducen a 256 dimensiones usando TruncatedSVD

### 2. Entrenamiento con AutoGluon
- Se entrenan modelos tabulares usando las características extraídas
- Se exploran diferentes algoritmos:
  - LightGBM (gradient boosting)
  - Modelos lineales
  - Ensambles ponderados
- Búsqueda de hiperparámetros con:
  - 5 trials
  - Scheduler local
  - Early stopping después de 25 rondas sin mejora

## Resultados Principales

### ResNet50
- Accuracy global: 86%
- Mejor desempeño en clases mayoritarias:
  - person (F1: 0.91)
  - car (F1: 0.91)
  - clutter (F1: 0.90)
- Desempeño moderado en clases minoritarias:
  - keyboard (F1: 0.92)
  - bookshelf (F1: 0.84)
  - table (F1: 0.24)

### MobileNetV2
- Accuracy global: 84%
- Buen desempeño en clases mayoritarias:
  - car (F1: 0.90)
  - keyboard (F1: 0.90)
  - clutter (F1: 0.88)
- Desempeño inferior en clases minoritarias:
  - table (F1: 0.27)
  - door (F1: 0.51)
  - building (F1: 0.55)

## Conclusiones
- ResNet50 ofrece mejor capacidad de representación, alcanzando métricas superiores
- MobileNetV2 proporciona un buen balance entre precisión y eficiencia computacional
- Las clases con más ejemplos y patrones visuales distintivos son mejor clasificadas
- El desbalance de clases afecta el desempeño en categorías minoritarias
- Los ensambles mejoran ligeramente el rendimiento sobre modelos individuales

## Requisitos
```python
pip install --no-cache-dir -U "numpy<2.0" "pandas<2.2" "autogluon==1.3.1"