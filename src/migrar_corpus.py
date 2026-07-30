import re
from datetime import datetime

import pandas as pd
import spacy
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from db_manager import get_collection, insertar_resenas_bulk

RUTA_CSV = "corpus_limpio.csv"
BORRAR_ANTES_DE_INSERTAR = True  # limpia la colección antes de subir el corpus

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def limpiar_texto(texto):
    """Minúsculas, quita dígitos y puntuación,/ñ."""
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = re.sub(r"[^a-záéíóúüñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def calcular_polaridad(calificacion):
    """1-2 negativa, 3 neutral, 4-5 positiva."""
    if calificacion is None:
        return None
    if calificacion <= 2:
        return "negativa"
    if calificacion == 3:
        return "neutral"
    return "positiva"


def calcular_metricas(doc_spacy):
    tokens = [t for t in doc_spacy if not t.is_punct and not t.is_space]
    num_palabras = len(tokens)

    if num_palabras == 0:
        return {
            "num_palabras": 0,
            "densidad_lexica": None,
            "ratio_sustantivos_verbos": None,
            "densidad_adjetivos": None,
        }

    conteo = {}
    for t in tokens:
        conteo[t.pos_] = conteo.get(t.pos_, 0) + 1

    num_contenido = sum(conteo.get(p, 0) for p in CONTENT_POS)
    num_noun = conteo.get("NOUN", 0)
    num_verb = conteo.get("VERB", 0)
    num_adj = conteo.get("ADJ", 0)

    return {
        "num_palabras": num_palabras,
        "densidad_lexica": round(num_contenido / num_palabras, 4),
        "ratio_sustantivos_verbos": round(num_noun / num_verb, 4) if num_verb else None,
        "densidad_adjetivos": round(num_adj / num_palabras, 4),
    }


def cargar_corpus(ruta_csv):
    df = pd.read_csv(ruta_csv)

    df["fecha"] = pd.to_datetime(df["fecha"], format="%m/%d/%Y", errors="coerce")
    df["fecha"] = df["fecha"].astype(object).where(df["fecha"].notna(), None)
    df["calificacion"] = pd.to_numeric(df["calificacion"], errors="coerce")

    print("Cargando modelo de spaCy...")
    nlp = spacy.load("es_core_news_md")

    textos_limpios = [limpiar_texto(t) for t in df["texto"]]
    docs_spacy = list(nlp.pipe(textos_limpios, batch_size=50))

    documentos = []
    for (_, fila), texto_limpio, doc_spacy in zip(df.iterrows(), textos_limpios, docs_spacy):
        tokens_nltk = word_tokenize(texto_limpio, preserve_line=True) if texto_limpio else []
        pos_nltk = [[w, tag] for w, tag in pos_tag(tokens_nltk)] if tokens_nltk else []
        pos_spacy = [[t.text, t.pos_, t.tag_, t.lemma_] for t in doc_spacy]

        calificacion = float(fila["calificacion"]) if pd.notna(fila["calificacion"]) else None

        documentos.append({
            "texto": fila["texto"],
            "texto_limpio": texto_limpio,
            "calificacion": calificacion,
            "tipo_lugar": fila.get("tipo_lugar"),
            "lugar": fila.get("lugar"),
            "fuente": fila.get("fuente"),
            "fecha": fila["fecha"],
            "fecha_recopilacion": datetime.utcnow(),
            "polaridad": calcular_polaridad(calificacion),
            "idioma": None,
            "url_fuente": None,
            "pos_tags": {
                "nltk": pos_nltk,
                "spacy": pos_spacy,
            },
            "embeddings": {
                "word2vec_avg": [],
                "beto_cls": [],
            },
            "metricas": calcular_metricas(doc_spacy),
        })

    return documentos


if __name__ == "__main__":
    coleccion = get_collection()

    total_previo = coleccion.count_documents({})
    print(f"Documentos actuales en la colección: {total_previo}")

    if BORRAR_ANTES_DE_INSERTAR:
        resultado_borrado = coleccion.delete_many({})
        print(f"Documentos borrados: {resultado_borrado.deleted_count}")

    documentos = cargar_corpus(RUTA_CSV)
    print(f"Filas leídas del CSV: {len(documentos)}")

    resultado = insertar_resenas_bulk(coleccion, documentos)
    print(f"Documentos insertados: {len(resultado.inserted_ids)}")

    pipeline = [{"$group": {"_id": "$tipo_lugar", "total": {"$sum": 1}}}]
    print("Conteo por tipo_lugar:", list(coleccion.aggregate(pipeline)))
