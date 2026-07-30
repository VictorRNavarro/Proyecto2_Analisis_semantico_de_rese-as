from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import UpdateOne
from gensim.models import Word2Vec

from db_manager import get_collection

BATCH_SIZE = 500
CARPETA = Path(__file__).resolve().parent

W2V_MODEL_PATH = CARPETA / "skipgram_model.bin"
BETO_NPY_PATH = CARPETA / "beto_embeddings.npy"
CORPUS_CSV_PATH = CARPETA / "corpus_limpio.csv"


def vector_promedio_w2v(tokens, wv):
    vectores = [wv[t] for t in tokens if t in wv]
    if not vectores:
        return np.zeros(wv.vector_size).tolist()
    return np.mean(vectores, axis=0).tolist()


def cargar_mapa_beto(ruta_npy, ruta_csv):
    vectores = np.load(ruta_npy)
    df = pd.read_csv(ruta_csv)

    mapa = {}
    for texto, vector in zip(df["texto"].tolist(), vectores):
        mapa.setdefault(texto, []).append(vector.tolist())
    return mapa


def main():
    coleccion = get_collection()

    wv = Word2Vec.load(str(W2V_MODEL_PATH)).wv
    mapa_beto = cargar_mapa_beto(BETO_NPY_PATH, CORPUS_CSV_PATH)
    usados_beto = {}

    docs = list(coleccion.find({}, {"texto": 1, "texto_limpio": 1}))
    total = len(docs)
    print(f"Documentos: {total}")

    operaciones = []
    sin_beto = 0

    for i, d in enumerate(docs, 1):
        texto = d.get("texto", "") or ""
        tokens = (d.get("texto_limpio") or texto).split()

        v_w2v = vector_promedio_w2v(tokens, wv)

        lista_beto = mapa_beto.get(texto)
        idx = usados_beto.get(texto, 0)
        if lista_beto and idx < len(lista_beto):
            v_beto = lista_beto[idx]
            usados_beto[texto] = idx + 1
        else:
            v_beto = None
            sin_beto += 1

        set_fields = {"embeddings.word2vec_avg": v_w2v}
        if v_beto is not None:
            set_fields["embeddings.beto_cls"] = v_beto

        operaciones.append(UpdateOne({"_id": d["_id"]}, {"$set": set_fields}))

        if i % BATCH_SIZE == 0:
            coleccion.bulk_write(operaciones)
            operaciones = []
            print(f"{i}/{total}")

    if operaciones:
        coleccion.bulk_write(operaciones)

    print(f"Listo. Sin match de BETO: {sin_beto}")


if __name__ == "__main__":
    main()
