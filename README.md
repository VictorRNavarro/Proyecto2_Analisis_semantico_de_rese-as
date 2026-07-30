# Proyecto 2: Análisis Semántico de Reseñas Turísticas de Costa Rica

**Curso:** Minería de Textos · Colegio Universitario de Cartago (CUC)
**Técnicas:** Bag of Words / TF-IDF, Word2Vec, BETO, MongoDB y Plotly Dash

## Objetivo

Analizar reseñas turísticas de Costa Rica para comparar tres representaciones de lenguaje: una representación dispersa basada en frecuencia (TF-IDF), embeddings estáticos entrenados en el corpus (Word2Vec) y embeddings contextuales en español (BETO). El proyecto incluye preparación del corpus, almacenamiento en MongoDB Atlas, análisis semántico y un dashboard interactivo.

## Corpus

El corpus local contiene **2,116 reseñas** de parques, hoteles y restaurantes. Cada registro incluye texto, calificación, tipo de lugar, fuente, fecha, lugar y polaridad.

> Los modelos y embeddings generados no se versionan por su tamaño. Después de clonar el repositorio se deben ejecutar los notebooks para regenerarlos.

## Estructura

```text
.
├── data/
│   ├── corpus.csv                         # Corpus base
│   ├── analysis/                          # Métricas y tablas generadas
│   ├── models/                            # Modelos Word2Vec y embeddings BETO (local)
│   └── visualizations/                    # Proyecciones y gráficos generados
├── dashboard/
│   └── app.py                             # Dashboard Plotly Dash
├── notebooks/
│   ├── 03_word2vec_analisis.ipynb         # CBOW, Skip-Gram y centroides
│   ├── 04_beto_analisis.ipynb             # BETO, polisemia, búsqueda y MLM
│   └── 05_comparacion_final.ipynb         # Comparación BoW, Word2Vec y BETO
├── scripts/
│   ├── run_polysemy_analysis.py           # Polisemia BETO a nivel de token
│   └── upgrade_notebooks.py                # Utilidad de actualización de notebooks
├── src/
│   ├── pipeline_completo_atlas.ipynb      # Pipeline de MongoDB Atlas
│   ├── scraper_resenas_costa_rica_api.py  # Recopilación de reseñas
│   ├── preprocessing.py                   # Limpieza y tokenización
│   ├── w2v_utils.py                       # Entrenamiento/análisis Word2Vec
│   └── beto_utils.py                      # Embeddings y análisis BETO
├── requirements.txt
└── USO_DE_IA.md
```

## Instalación

Requiere Python 3.10 o superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download es_core_news_md
```

Para recursos de NLTK:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## Ejecución

Ejecutar desde la carpeta raíz del repositorio.

### 1. MongoDB Atlas y scraping

Configurar la cadena de conexión de MongoDB Atlas y las variables necesarias en `src/pipeline_completo_atlas.ipynb`. El notebook realiza la carga del corpus, procesamiento y almacenamiento de documentos.

Para ejecutar el scraper con una clave configurada como variable de entorno:

```powershell
$env:SERPAPI_KEY="TU_CLAVE"
.\.venv\Scripts\python.exe src\scraper_resenas_costa_rica_api.py
```

No se deben guardar claves API en el código ni subirlas al repositorio.

### 2. Análisis semántico

Ejecutar los notebooks en este orden:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\03_word2vec_analisis.ipynb
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\04_beto_analisis.ipynb
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\05_comparacion_final.ipynb
```

El notebook de BETO descarga el modelo `dccuchile/bert-base-spanish-wwm-cased` la primera vez. Puede tardar varios minutos en CPU; se recomienda GPU cuando esté disponible.

### 3. Dashboard

```powershell
.\.venv\Scripts\python.exe dashboard\app.py
```

Abrir [http://127.0.0.1:8050](http://127.0.0.1:8050).

El dashboard contiene:

- Resumen del corpus y distribución de calificaciones.
- Proyección Word2Vec, palabras similares, centroides y vocabulario distintivo.
- Búsqueda semántica, t-SNE y polisemia contextual con BETO.
- Comparación de clasificación, clustering y t-SNE de las tres representaciones.

## Metodología

| Representación | Idea principal | Uso en el proyecto |
|---|---|---|
| TF-IDF | Pondera las palabras importantes de cada reseña. | Línea base para clasificación y clustering. |
| Word2Vec | Aprende vectores estáticos según coocurrencia. | Similitudes, analogías, centroides y embeddings promedio de reseñas. |
| BETO | Genera vectores contextuales para español. | Polisemia, búsqueda semántica, MLM y clasificación. |

La similitud entre vectores se calcula principalmente con **similitud coseno**. Para las categorías se usan centroides, es decir, el promedio de los vectores de sus reseñas.

## Resultados principales

| Tarea | TF-IDF | Word2Vec | BETO |
|---|---:|---:|---:|
| Clasificación de polaridad (accuracy) | 89.39% | 90.33% | 89.62% |
| Clasificación de tipo de lugar (accuracy) | 88.21% | 91.04% | 91.51% |

El clustering obtuvo valores bajos de Silhouette Score. Este resultado se interpreta con cautela: el lenguaje de las reseñas turísticas comparte términos entre parques, hoteles y restaurantes, por lo que las categorías no se separan de forma clara sin supervisión.

El corpus está desbalanceado hacia reseñas positivas y parques. Por ello, la accuracy debe complementarse con F1 macro, métricas por clase y matrices de confusión al interpretar los resultados.

## Integrantes

- Steven Vindas Rivera
- Victor Rojas Navarro 

## Uso de IA

El uso de herramientas de IA y las modificaciones realizadas se documentan en [USO_DE_IA.md](USO_DE_IA.md).
