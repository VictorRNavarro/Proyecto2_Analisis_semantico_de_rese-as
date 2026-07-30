"""Add the missing rubric analyses to the project notebooks."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def append_cell(notebook_path: Path, source: str, marker: str) -> None:
    notebook = nbf.read(notebook_path, as_version=4)
    if any(marker in "".join(cell.source) for cell in notebook.cells):
        notebook.nbformat_minor = max(notebook.nbformat_minor, 5)
        nbf.write(notebook, notebook_path)
        return
    notebook.nbformat_minor = max(notebook.nbformat_minor, 5)
    notebook.cells.extend([
        nbf.v4.new_markdown_cell(marker),
        nbf.v4.new_code_cell(source),
    ])
    nbf.write(notebook, notebook_path)


def enable_beto_cache(notebook_path: Path) -> None:
    """Avoid regenerating 2k embeddings when the verified cache already exists."""
    notebook = nbf.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        source = "".join(cell.source)
        if "beto_embeddings = embedder.embed_batch(" not in source:
            continue
        cell.source = '''# Generar embeddings BETO o reutilizar el caché verificable
from pathlib import Path

cache_embeddings = Path("../data/models/beto_embeddings.npy")
if cache_embeddings.exists():
    beto_embeddings = load_embeddings(str(cache_embeddings))
    if len(beto_embeddings) != len(df):
        raise ValueError("El caché BETO no coincide con el corpus actual; elimínalo y vuelve a generarlo.")
    print(f"✓ Embeddings BETO cargados desde caché: {beto_embeddings.shape}")
else:
    print(f"Generando embeddings para {len(df)} reseñas...")
    beto_embeddings = embedder.embed_batch(df["texto"].tolist(), use_cls=True, batch_size=32)
    save_embeddings(beto_embeddings, str(cache_embeddings))
    print(f"✓ Embeddings BETO generados: {beto_embeddings.shape}")
'''
        break
    notebook.nbformat_minor = max(notebook.nbformat_minor, 5)
    nbf.write(notebook, notebook_path)


append_cell(
    ROOT / "notebooks" / "03_word2vec_analisis.ipynb",
    '''# Centroides por tipo de lugar y polaridad + vocabulario característico
from sklearn.metrics.pairwise import cosine_similarity

def embedding_documento(tokens, wv):
    vectores = [wv[token] for token in tokens if token in wv]
    return np.mean(vectores, axis=0) if vectores else np.zeros(wv.vector_size)

df_centroides = df.copy()
df_centroides["tokens_w2v"] = tokenized_corpus
df_centroides["embedding_w2v"] = [embedding_documento(tokens, skipgram_analyzer.wv) for tokens in tokenized_corpus]

grupos = {}
for columna in ["tipo_lugar", "polaridad"]:
    for categoria, subset in df_centroides.groupby(columna):
        grupos[f"{columna}:{categoria}"] = np.mean(np.vstack(subset["embedding_w2v"]), axis=0)

nombres_grupos = list(grupos)
matriz_centroides = cosine_similarity(np.vstack([grupos[nombre] for nombre in nombres_grupos]))
df_centroid_similarity = pd.DataFrame(matriz_centroides, index=nombres_grupos, columns=nombres_grupos)
display(df_centroid_similarity.round(3))
df_centroid_similarity.to_csv("../data/analysis/w2v_centroid_similarities.csv")

filas_vocabulario = []
for nombre, centroide in grupos.items():
    otros = [grupos[otro] for otro in nombres_grupos if otro != nombre and otro.split(":")[0] == nombre.split(":")[0]]
    candidatos = []
    for palabra in skipgram_analyzer.wv.index_to_key:
        vector = skipgram_analyzer.wv[palabra]
        propio = cosine_similarity([vector], [centroide])[0, 0]
        contraste = max((cosine_similarity([vector], [otro])[0, 0] for otro in otros), default=0)
        candidatos.append((palabra, propio, propio - contraste))
    for rango, (palabra, similitud, distintividad) in enumerate(sorted(candidatos, key=lambda x: x[2], reverse=True)[:15], 1):
        filas_vocabulario.append({"grupo": nombre, "rango": rango, "palabra": palabra, "similitud_centroide": similitud, "distintividad": distintividad})

df_vocabulario_caracteristico = pd.DataFrame(filas_vocabulario)
display(df_vocabulario_caracteristico)
df_vocabulario_caracteristico.to_csv("../data/analysis/w2v_characteristic_vocabulary.csv", index=False)
print("Se guardaron centroides y vocabulario distintivo por tipo de lugar y polaridad.")
''',
    "## 8. Centroides y vocabulario característico",
)

append_cell(
    ROOT / "notebooks" / "04_beto_analisis.ipynb",
    '''# Polisemia contextual a nivel del token, no a nivel de oración [CLS]
from sklearn.metrics.pairwise import cosine_similarity

palabras_polisemicas = ["rico", "fresco", "caro", "vista", "ambiente"]
filas_polisemia = []
for palabra in palabras_polisemicas:
    contextos = df[df["texto"].str.contains(rf"\\b{palabra}\\b", case=False, regex=True, na=False)]["texto"].head(5).tolist()
    vectores = []
    contextos_validos = []
    for contexto in contextos:
        vector = embedder.get_word_embedding(contexto, palabra)
        if vector is not None:
            vectores.append(vector)
            contextos_validos.append(contexto)
    if len(vectores) >= 2:
        similitudes = cosine_similarity(np.vstack(vectores))
        for i in range(len(contextos_validos)):
            for j in range(i + 1, len(contextos_validos)):
                filas_polisemia.append({
                    "palabra": palabra,
                    "contexto_1": contextos_validos[i][:220],
                    "contexto_2": contextos_validos[j][:220],
                    "similitud_token_contextual": similitudes[i, j],
                })
    else:
        filas_polisemia.append({"palabra": palabra, "contexto_1": "No hay dos contextos válidos en el corpus", "contexto_2": "", "similitud_token_contextual": np.nan})

df_polisemia = pd.DataFrame(filas_polisemia)
display(df_polisemia)
df_polisemia.to_csv("../data/analysis/beto_polysemy_token_analysis.csv", index=False)
print("Interpretación: similitudes menores entre contextos muestran que BETO cambia la representación de la palabra según el uso contextual.")
''',
    "## 12. Polisemia contextual a nivel de token",
)

append_cell(
    ROOT / "notebooks" / "05_comparacion_final.ipynb",
    '''# Requisito adicional: clasificación por tipo de lugar y coherencia de clusters
from sklearn.metrics import adjusted_rand_score, homogeneity_score
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

y_lugar = LabelEncoder().fit_transform(df["tipo_lugar"])
idx_train_lugar, idx_test_lugar = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y_lugar)
resultados_lugar = []
for nombre, embeddings in [("TF-IDF", bow_embeddings), ("Word2Vec", w2v_embeddings), ("BETO", beto_embeddings)]:
    clasificador = LogisticRegression(random_state=42, max_iter=1500)
    clasificador.fit(embeddings[idx_train_lugar], y_lugar[idx_train_lugar])
    predicciones = clasificador.predict(embeddings[idx_test_lugar])
    resultados_lugar.append({"Método": nombre, "Tarea": "tipo_lugar", "Accuracy": accuracy_score(y_lugar[idx_test_lugar], predicciones), "F1_macro": f1_score(y_lugar[idx_test_lugar], predicciones, average="macro", zero_division=0)})
df_clasificacion_lugar = pd.DataFrame(resultados_lugar)
display(df_clasificacion_lugar)
df_clasificacion_lugar.to_csv("../data/analysis/classification_place_type_metrics.csv", index=False)

alineacion_clusters = []
for nombre, embeddings in [("TF-IDF", bow_embeddings), ("Word2Vec", w2v_embeddings), ("BETO", beto_embeddings)]:
    etiquetas_cluster = KMeans(n_clusters=df["tipo_lugar"].nunique(), random_state=42, n_init=10).fit_predict(embeddings)
    alineacion_clusters.append({"Método": nombre, "n_clusters": df["tipo_lugar"].nunique(), "Silhouette": silhouette_score(embeddings, etiquetas_cluster), "ARI_tipo_lugar": adjusted_rand_score(y_lugar, etiquetas_cluster), "Homogeneidad_tipo_lugar": homogeneity_score(y_lugar, etiquetas_cluster)})
df_alineacion_clusters = pd.DataFrame(alineacion_clusters)
display(df_alineacion_clusters)
df_alineacion_clusters.to_csv("../data/analysis/clustering_category_alignment.csv", index=False)

# t-SNE comparable: mismo subconjunto y mismas etiquetas para las tres representaciones
indices_tsne = np.random.default_rng(42).choice(len(df), size=min(800, len(df)), replace=False)
filas_tsne = []
for nombre, embeddings in [("TF-IDF", bow_embeddings), ("Word2Vec", w2v_embeddings), ("BETO", beto_embeddings)]:
    coordenadas = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=750, init="pca").fit_transform(embeddings[indices_tsne])
    for indice, (x, y) in zip(indices_tsne, coordenadas):
        filas_tsne.append({"metodo": nombre, "x": x, "y": y, "tipo_lugar": df.iloc[indice]["tipo_lugar"], "polaridad": df.iloc[indice]["polaridad"], "texto": df.iloc[indice]["texto"][:160]})
df_tsne_comparativo = pd.DataFrame(filas_tsne)
df_tsne_comparativo.to_csv("../data/analysis/tsne_comparison.csv", index=False)
display(df_tsne_comparativo.head())
print("Conclusión responsable: compare los resultados observados; no asuma que BETO gana si las métricas no lo respaldan.")
''',
    "## 12. Clasificación por tipo y evaluación comparativa de clusters",
)

enable_beto_cache(ROOT / "notebooks" / "04_beto_analisis.ipynb")

print("Notebooks actualizados.")
