"""Generate BETO token-level polysemy evidence from cached project artifacts."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    with open(ROOT / "data" / "models" / "beto_analyzer.pkl", "rb") as file:
        analyzer = pickle.load(file)
    df = pd.read_csv(ROOT / "data" / "corpus.csv")

    rows = []
    for word in ["rico", "fresco", "caro", "vista", "ambiente"]:
        contexts = df[df["texto"].str.contains(rf"\b{word}\b", case=False, regex=True, na=False)]["texto"].head(5).tolist()
        vectors, valid_contexts = [], []
        for context in contexts:
            vector = analyzer.embedder.get_word_embedding(context, word)
            if vector is not None:
                vectors.append(vector)
                valid_contexts.append(context)
        if len(vectors) < 2:
            rows.append({"palabra": word, "contexto_1": "No hay dos contextos válidos en el corpus", "contexto_2": "", "similitud_token_contextual": np.nan})
            continue
        similarities = cosine_similarity(np.vstack(vectors))
        for first in range(len(valid_contexts)):
            for second in range(first + 1, len(valid_contexts)):
                rows.append({
                    "palabra": word,
                    "contexto_1": valid_contexts[first][:220],
                    "contexto_2": valid_contexts[second][:220],
                    "similitud_token_contextual": similarities[first, second],
                })

    output = ROOT / "data" / "analysis" / "beto_polysemy_token_analysis.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Análisis guardado en {output}")


if __name__ == "__main__":
    main()
