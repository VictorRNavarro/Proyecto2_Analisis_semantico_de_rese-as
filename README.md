# Proyecto 2: Análisis Semántico de Reseñas Turísticas de Costa Rica

**Curso:** Minería de Textos - CUC  
**Tema:** Análisis de Representaciones Semánticas (BoW, Word2Vec, BETO)  
**Duración estimada:** 8-10 semanas  

---

## 📋 Descripción del Proyecto

Este proyecto aplica técnicas avanzadas de **Procesamiento de Lenguaje Natural (PLN)** para analizar **~2,100 reseñas de turismo en Costa Rica** usando tres enfoques:

1. **Bag of Words (BoW) / TF-IDF**: Línea base con vectores dispersos
2. **Word2Vec**: Embeddings estáticos densos aprendidos del corpus
3. **BETO**: Embeddings contextuales profundos (Transformer - Spanish BERT)

**Objetivo final:** Comparar estas técnicas en tareas de:
- Similitud semántica
- Clasificación de reseñas
- Clustering automático
- Búsqueda semántica

---

## 🎯 Objetivos Específicos

- [ ] **Scraping & MongoDB (20 pts):** Migrar corpus a base de datos NoSQL
- [ ] **Word2Vec (25 pts):** Entrenar modelos CBOW y Skip-Gram, análisis semántico
- [ ] **BETO (25 pts):** Embeddings contextuales, búsqueda semántica, polisemia
- [ ] **Comparación (15 pts):** Métricas cuantitativas (clasificación, clustering)
- [ ] **Dashboard (10 pts):** Visualizaciones interactivas con Plotly Dash
- [ ] **GitHub & Documentación (5 pts):** Repositorio limpio, USO_DE_IA.md

---

## 📂 Estructura del Proyecto

```
proyecto2-analisis-semantico-resenas/
│
├── README.md                           # Este archivo
├── USO_DE_IA.md                        # Documentación de uso de IA
├── requirements.txt                    # Dependencias Python
├── .gitignore
│
├── notebooks/
│   ├── 01_migracion_mongodb.ipynb     # (Hecho por otro integrante)
│   ├── 02_web_scraping.ipynb          # (Hecho por otro integrante)
│   ├── 03_word2vec_analisis.ipynb     # Word2Vec entrenamiento
│   ├── 04_beto_analisis.ipynb         # BETO y análisis contextual
│   └── 05_comparacion_final.ipynb     # Comparación de métodos
│
├── src/
│   ├── preprocessing.py                # Limpieza y tokenización
│   ├── w2v_utils.py                   # Utilidades Word2Vec
│   ├── beto_utils.py                  # Utilidades BETO
│   └── dashboard_utils.py             # Funciones para dashboard
│
├── dashboard/
│   └── app.py                         # Dashboard Plotly Dash
│
├── data/
│   ├── corpus_limpio.csv              # Reseñas (2,116 documentos)
│   ├── models/
│   │   ├── cbow_model.bin             # Modelo CBOW entrenado
│   │   ├── skipgram_model.bin         # Modelo Skip-Gram entrenado
│   │   ├── cbow_analyzer.pkl          # Analizador CBOW
│   │   ├── skipgram_analyzer.pkl      # Analizador Skip-Gram
│   │   ├── beto_embeddings.npy        # Embeddings BETO cachados
│   │   ├── beto_analyzer.pkl          # Analizador BETO
│   │   └── beto_cluster_labels.npy    # Labels de clustering
│   │
│   ├── visualizations/
│   │   ├── tsne_vectors_w2v.npy       # t-SNE Word2Vec (2D)
│   │   ├── tsne_words_w2v.pkl         # Palabras para t-SNE
│   │   ├── tsne_categories_w2v.pkl    # Categorías semánticas
│   │   ├── tsne_vectors_beto.npy      # t-SNE BETO (2D)
│   │   └── ...
│   │
│   └── analysis/
│       ├── w2v_similarities.csv        # Similitudes Word2Vec
│       ├── w2v_vocabulary_by_place.csv # Vocabulario por tipo
│       ├── beto_tsne_data.csv         # Datos t-SNE BETO
│       ├── beto_cluster_summary.csv   # Resumen de clusters
│       └── ...
│
└── .github/
    └── workflows/
        └── ci.yml                     # CI/CD (opcional)
```

---

## 🚀 Instalación y Setup

### Requisitos Previos

- **Python 3.9+** 
- **pip** o **conda**
- **Git**
- **GPU (NVIDIA)** recomendada para BETO (opcional, funciona en CPU)

### Paso 1: Clonar Repositorio

```bash
git clone https://github.com/tu-usuario/proyecto2-analisis-semantico.git
cd proyecto2-analisis-semantico
```

### Paso 2: Crear Entorno Virtual

```bash
# Con venv
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# O con conda
conda create -n proyecto2 python=3.10
conda activate proyecto2
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt

# Descargar modelos de spaCy
python -m spacy download es_core_news_sm

# Descargar recursos NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Paso 4: Crear Directorios

```bash
mkdir -p data/{models,visualizations,analysis}
```

### Paso 5: Copiar Corpus

Coloca `corpus_limpio.csv` en `data/`:

```bash
cp corpus_limpio.csv data/
```

---

## 📊 Ejecución de Análisis

### A. Word2Vec (30-45 minutos)

```bash
jupyter notebook notebooks/03_word2vec_analisis.ipynb
```

**Qué hace:**
- Carga corpus y preprocesa
- Entrena CBOW (10 épocas)
- Entrena Skip-Gram (10 épocas)
- Analiza similitudes y analogías
- Genera visualización t-SNE
- Guarda modelos y análisis

**Salida:**
- `data/models/cbow_model.bin` (110 MB aprox)
- `data/models/skipgram_model.bin` (110 MB aprox)
- `data/analysis/w2v_similarities.csv`

### B. BETO (15-30 minutos en GPU, 1-2 horas en CPU)

```bash
jupyter notebook notebooks/04_beto_analisis.ipynb
```

**Qué hace:**
- Carga BETO desde HuggingFace
- Genera embeddings para todas las reseñas
- Analiza polisemia contextual
- Realiza búsqueda semántica
- Clustering con silhouette score
- Masked Language Model predicciones
- Visualizaciones t-SNE

**Salida:**
- `data/models/beto_embeddings.npy` (500 MB aprox)
- `data/analysis/beto_cluster_summary.csv`

### C. Comparación Final (15 minutos)

```bash
jupyter notebook notebooks/05_comparacion_final.ipynb
```

**Qué hace:**
- Carga todos los modelos
- Clasifica reseñas con cada método
- Calcula métricas: Accuracy, Precision, Recall
- Clustering con KMeans (cada método)
- Calcula Silhouette Scores
- Genera gráficos comparativos

**Salida:**
- `data/analysis/comparison_metrics.csv`
- Visualizaciones de rendimiento

---

## 🎨 Ejecutar Dashboard

```bash
python dashboard/app.py
```

Luego abre: **http://localhost:8050**

### Funcionalidades del Dashboard

| Tab | Descripción | Interactividad |
|---|---|---|
| 📊 Overview | Distribución de calificaciones y tipos | Filtros por fecha |
| 🔤 Word2Vec | t-SNE de palabras, similitudes | Hover para ver detalles |
| 🧠 BETO | Búsqueda semántica, t-SNE reseñas | **Buscador en tiempo real** |
| ⚖️ Comparación | Tabla comparativa de métodos | Sorting y export CSV |

---

## 📈 Resultados Esperados

### Word2Vec

**Vocabulario:** ~5,000-7,000 palabras únicas  
**Dimensiones:** 300  

**Ejemplo de Similitudes:**
```
hotel ↔ hospedaje: 0.78
volcán ↔ naturaleza: 0.65
guía ↔ conductor: 0.72
recomendado ↔ excelente: 0.81
```

**Ejemplo de Analogías:**
```
hotel → habitación como restaurante → ???
Predicción: "comedor" (similitud: 0.62)

volcán → naturaleza como playa → ???
Predicción: "mar" (similitud: 0.58)
```

### BETO

**Dimensiones:** 768  
**Búsqueda Semántica (top-1 accuracy):** ~75-85%  
**Clustering (Silhouette Score):** 0.40-0.55 (corpus pequeño)  

**Ejemplo de Polisemia:**
```
"El banco de la orilla es bonito"          → embedding_1
"El banco de dinero es confiable"          → embedding_2
Similitud: 0.45 (contextos diferentes)
```

### Comparación

| Métrica | BoW | Word2Vec | BETO |
|---|---|---|---|
| Accuracy (clasificación polaridad) | 72% | 78% | 85% |
| Silhouette Score (clustering) | 0.32 | 0.38 | 0.52 |
| Velocidad (1000 docs) | <1s | 5s | 120s |
| Memoria (1000 docs) | 50 MB | 200 MB | 500 MB |

---

## 📝 Tareas por Integrante

### Tu Responsabilidad (Word2Vec, BETO, Dashboard)

- [x] Cargar corpus_limpio.csv
- [ ] Ejecutar notebook Word2Vec (03_word2vec_analisis.ipynb)
- [ ] Ejecutar notebook BETO (04_beto_analisis.ipynb)
- [ ] Ejecutar notebook Comparación (05_comparacion_final.ipynb)
- [ ] Probar dashboard en local
- [ ] Hacer commits significativos en GitHub
- [ ] Documentar cualquier bug encontrado
- [ ] Actualizar USO_DE_IA.md con tus prompts (si usas IA)

### Responsabilidad del Otro Integrante

- [ ] Scraping (01 - Web scraping)
- [ ] MongoDB (02 - Migración a base de datos)
- [ ] Validación de datos combinados

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'gensim'"
```bash
pip install gensim --upgrade
```

### Error: "CUDA out of memory" en BETO
```python
# En beto_utils.py, reducir batch_size:
beto_embeddings = embedder.embed_batch(
    texts,
    batch_size=16  # Reducir de 32 a 16
)
```

### BETO muy lento en CPU
**Solución esperada:** Usar GPU (NVIDIA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### t-SNE no converge
```python
# En notebooks, aumentar iteraciones:
tsne = TSNE(n_components=2, random_state=42, n_iter=2000)
```

---

## 📚 Referencias y Recursos

### Librerías Utilizadas

- **gensim:** [Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html)
- **transformers:** [BETO docs](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)
- **scikit-learn:** [Clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
- **plotly:** [Dash tutorial](https://dash.plotly.com/layout)

### Papers Relevantes

- [Efficient Estimation of Word Representations (Word2Vec)](https://arxiv.org/abs/1301.3781)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [BETO: Spanish BERT (Paper)](https://github.com/dccuchile/beto)

### Recursos de Semántica

- [The Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics)
- [Cosine Similarity en NLP](https://en.wikipedia.org/wiki/Cosine_similarity)
- [t-SNE: Visualization of High-Dimensional Data](https://distill.pub/2016/misread-tsne/)

---

## ✅ Checklist Antes de Entregar

### Código

- [ ] Todos los notebooks ejecutan sin errores
- [ ] Modelos se guardan correctamente en `data/models/`
- [ ] Dashboard carga sin problemas en localhost:8050
- [ ] requirements.txt incluye todas las librerías necesarias
- [ ] No hay rutas hardcodeadas (todo relativo)

### Documentación

- [ ] README.md está completo y actualizado
- [ ] USO_DE_IA.md documenta prompts utilizados
- [ ] Cada notebook tiene markdown explicativo
- [ ] Docstrings en todas las funciones (src/*.py)

### GitHub

- [ ] Repositorio público creado
- [ ] Mínimo 15 commits significativos
- [ ] `.gitignore` excluye archivos grandes (*.bin, *.npy)
- [ ] Branch main limpio, sin archivos de prueba

### Análisis

- [ ] Word2Vec: 5+ analogías documentadas
- [ ] BETO: 3+ ejemplos de polisemia
- [ ] Comparación: tabla con 4+ métricas
- [ ] Dashboard: todos los tabs funcionan

---

## 📞 Soporte

### Preguntas Frecuentes

**P: ¿Cuánto tiempo tarda entrenar Word2Vec?**  
R: ~30-45 minutos en CPU de 8 núcleos. CBOW es más rápido que Skip-Gram.

**P: ¿Se puede usar CPU para BETO?**  
R: Sí, pero tarda ~1-2 horas. GPU (NVIDIA) lo hace en 5-15 minutos.

**P: ¿Qué significa Silhouette Score?**  
R: Métrica de clustering. Rango [-1, 1]. Valores >0.5 son buenos clustering.

**P: ¿Cómo hago búsqueda semántica en el dashboard?**  
R: En el tab "BETO", escribe en el buscador: "Quiero ver naturaleza" y presiona Buscar.

---

## 📄 Licencia

Este proyecto es para fines educativos en el curso Minería de Textos - CUC.

---

## 👥 Autores

- **Integrante 1:** [Tu nombre] - Word2Vec, BETO, Dashboard
- **Integrante 2:** [Nombre del compañero] - Scraping, MongoDB

**Última actualización:** 2024  
**Estado:** En desarrollo ✏️

---

## Notas de Implementación

### Performance Tips

1. **Cachear BETO:** Después de generar embeddings, no regeneres (son 500 MB)
2. **Usar GPU para BETO:** 10x más rápido que CPU
3. **Batch processing:** Procesar de a 32 documentos ahorra memoria
4. **t-SNE:** Sample de 1000 docs en lugar de todas (visualización más clara)

### Próximas Mejoras (Opcional)

- [ ] Agregar búsqueda por filtros (calificación, tipo_lugar)
- [ ] Exportar resultados a PDF
- [ ] Comparar con otros modelos (FastText, RoBERTa)
- [ ] Análisis de sentimientos con BETO
- [ ] Deploy en Heroku/AWS

---

**¡Éxito en el proyecto!** 🚀
