"""Dashboard Plotly Dash para el Proyecto 2.

Se ejecuta con ``python dashboard/app.py`` desde la raiz del repositorio.
Los analisis entrenados son opcionales: cuando faltan, la interfaz explica
como generarlos sin impedir que el dashboard abra.
"""

from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd
import dash
from dash import Dash, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
VIZ_DIR = DATA_DIR / "visualizations"
ANALYSIS_DIR = DATA_DIR / "analysis"

# Paleta inspirada en Costa Rica: bosque tropical, océano, arena y coral.
COLORS = {
    "forest": "#176B4D",
    "leaf": "#2E9B6D",
    "ocean": "#087E8B",
    "sky": "#BFE8F2",
    "sand": "#FFF4DD",
    "coral": "#E76F51",
    "ink": "#183B3A",
    "cream": "#FAFCF7",
}
PLACE_COLORS = {"parque": COLORS["leaf"], "hotel": COLORS["ocean"], "restaurante": COLORS["coral"]}


def load_corpus() -> pd.DataFrame:
    """Carga el corpus aceptando el nombre solicitado y el nombre actual."""
    for path in (DATA_DIR / "corpus_limpio.csv", DATA_DIR / "corpus.csv"):
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No se encontro data/corpus.csv ni data/corpus_limpio.csv")


def optional_csv(name: str):
    path = ANALYSIS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def optional_npy(folder: Path, name: str):
    path = folder / name
    return np.load(path) if path.exists() else None


df = load_corpus()
df["calificacion"] = pd.to_numeric(df["calificacion"], errors="coerce")
beto_embeddings = optional_npy(MODELS_DIR, "beto_embeddings.npy")
tsne_w2v = optional_npy(VIZ_DIR, "tsne_vectors_w2v.npy")
tsne_beto = optional_npy(VIZ_DIR, "tsne_vectors_beto.npy")
w2v_similarity = optional_csv("w2v_similarities.csv")
w2v_vocabulary = optional_csv("w2v_vocabulary_by_place.csv")
w2v_centroid_similarity = optional_csv("w2v_centroid_similarities.csv")
w2v_characteristic_vocabulary = optional_csv("w2v_characteristic_vocabulary.csv")
beto_tsne = optional_csv("beto_tsne_data.csv")
beto_polysemy = optional_csv("beto_polysemy_token_analysis.csv")
classification_metrics = optional_csv("classification_metrics.csv")
classification_place_type_metrics = optional_csv("classification_place_type_metrics.csv")
clustering_metrics = optional_csv("clustering_metrics.csv")
clustering_category_alignment = optional_csv("clustering_category_alignment.csv")
comparison_features = optional_csv("comparison_features.csv")
tsne_comparison = optional_csv("tsne_comparison.csv")

if beto_tsne is not None and {"x", "y"}.issubset(beto_tsne.columns):
    beto_tsne = beto_tsne.copy()


def warning(message: str):
    return dbc.Alert(message, color="secondary", className="mt-3")


def figure_message(message: str):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 16})
    fig.update_layout(height=420, template="plotly_white", paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"], font={"color": COLORS["ink"]})
    return fig


def tourist_figure(figure):
    """Aplica un estilo consistente a todas las visualizaciones Plotly."""
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["cream"],
        plot_bgcolor=COLORS["cream"],
        font={"color": COLORS["ink"], "family": "Arial"},
        title={"font": {"color": COLORS["forest"], "size": 20}},
        margin={"l": 45, "r": 30, "t": 65, "b": 45},
    )
    return figure


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Reseñas turísticas: Word2Vec y BETO"

stats = [
    ("Reseñas", f"{len(df):,}"),
    ("Calificación media", f"{df['calificacion'].mean():.2f}"),
    ("Tipos de lugar", str(df["tipo_lugar"].nunique())),
    ("Polaridades", str(df["polaridad"].nunique())),
]

app.layout = dbc.Container(
    [
        dbc.Card(
            dbc.CardBody([
                html.H1("Costa Rica en palabras", className="mb-1", style={"color": "white", "fontWeight": "700"}),
                html.P("Análisis semántico de reseñas turísticas · Word2Vec · BETO", className="mb-0", style={"color": "#E5F7EC", "fontSize": "1.1rem"}),
            ]),
            className="mt-4 mb-3 shadow-sm",
            style={"background": f"linear-gradient(120deg, {COLORS['forest']}, {COLORS['ocean']})", "border": "0", "borderRadius": "16px"},
        ),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Small(k, style={"color": COLORS["forest"], "fontWeight": "600"}), html.H3(v, style={"color": COLORS["ink"], "fontWeight": "700"})]), className="shadow-sm h-100", style={"borderTop": f"5px solid {color}", "borderRadius": "12px"}), md=3, className="mb-3")
            for (k, v), color in zip(stats, [COLORS["leaf"], COLORS["ocean"], COLORS["coral"], "#D4A017"])
        ]),
        dcc.Tabs(
            id="tab",
            value="overview",
            className="mt-4",
            colors={"border": COLORS["sky"], "primary": COLORS["forest"], "background": "white"},
            children=[
                dcc.Tab(label="🌿 Resumen", value="overview"),
                dcc.Tab(label="🧭 Word2Vec", value="word2vec"),
                dcc.Tab(label="🦜 BETO", value="beto"),
                dcc.Tab(label="🌊 Comparación", value="comparison"),
            ],
        ),
        html.Div(id="tab-content", className="py-3"),
        html.Footer("Proyecto 2 · Minería de Textos · Turismo de Costa Rica", className="text-center py-3", style={"color": COLORS["forest"], "fontWeight": "600"}),
    ],
    fluid=True,
    style={"backgroundColor": COLORS["cream"], "minHeight": "100vh", "paddingLeft": "3%", "paddingRight": "3%"},
)


def overview_layout():
    ratings = df["calificacion"].value_counts().sort_index().reset_index()
    ratings.columns = ["calificacion", "cantidad"]
    fig_ratings = tourist_figure(px.bar(ratings, x="calificacion", y="cantidad", title="Distribución de calificaciones", color_discrete_sequence=[COLORS["ocean"]]))
    fig_places = tourist_figure(px.pie(df, names="tipo_lugar", title="Reseñas por tipo de lugar", color="tipo_lugar", color_discrete_map=PLACE_COLORS))
    fig_average = tourist_figure(px.bar(df.groupby("tipo_lugar", as_index=False)["calificacion"].mean(), x="tipo_lugar", y="calificacion", color="tipo_lugar", title="Calificación media por tipo", color_discrete_map=PLACE_COLORS))
    return dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_ratings), md=6),
        dbc.Col(dcc.Graph(figure=fig_places), md=6),
        dbc.Col(dcc.Graph(figure=fig_average), md=12),
    ])


def word2vec_layout():
    components = []
    if tsne_w2v is not None:
        points = pd.DataFrame(tsne_w2v[:, :2], columns=["x", "y"])
        components.append(dcc.Graph(figure=tourist_figure(px.scatter(points, x="x", y="y", title="Word2Vec: proyección t-SNE de palabras", color_discrete_sequence=[COLORS["leaf"]]))))
    else:
        components.append(warning("Ejecuta notebooks/03_word2vec_analisis.ipynb para generar la proyección t-SNE y las tablas de análisis."))
    for title, table in [("Vecinos semánticos y analogías", w2v_similarity), ("Vocabulario característico", w2v_vocabulary)]:
        if table is not None:
            components.extend([html.H4(title), dbc.Table.from_dataframe(table, striped=True, bordered=True, hover=True, responsive=True)])
    if w2v_centroid_similarity is not None:
        components.extend([html.H4("Similitud entre centroides: tipo de lugar y polaridad"), dbc.Table.from_dataframe(w2v_centroid_similarity.round(3), striped=True, bordered=True, hover=True, responsive=True)])
    if w2v_characteristic_vocabulary is not None:
        components.extend([html.H4("Palabras distintivas por grupo"), dbc.Table.from_dataframe(w2v_characteristic_vocabulary.round(3), striped=True, bordered=True, hover=True, responsive=True)])
    return html.Div(components)


def beto_layout():
    graph = figure_message("Ejecuta notebooks/04_beto_analisis.ipynb para generar los embeddings BETO.")
    if beto_tsne is not None:
        color = "tipo_lugar" if "tipo_lugar" in beto_tsne else None
        graph = tourist_figure(px.scatter(beto_tsne, x="x", y="y", color=color, hover_data=[c for c in ["texto", "calificacion"] if c in beto_tsne], title="BETO: proyección t-SNE de reseñas", color_discrete_map=PLACE_COLORS))
    components = [
        dbc.InputGroup([dbc.Input(id="query", placeholder="Ej.: un lugar tranquilo rodeado de naturaleza"), dbc.Button("Buscar", id="search", color="primary")]),
        html.Div(id="search-results", className="mt-3"),
        dcc.Graph(figure=graph, className="mt-3"),
    ]
    if beto_polysemy is not None:
        components.extend([
            html.H4("Polisemia contextual a nivel de token", className="mt-4"),
            html.P("Una similitud menor entre contextos indica que BETO representó la misma palabra de forma más diferente según su contexto.", className="text-muted"),
            dbc.Table.from_dataframe(beto_polysemy.round(3), striped=True, bordered=True, hover=True, responsive=True),
        ])
    return html.Div(components)


def comparison_layout():
    sections = []
    for title, table in [
        ("Características de las representaciones", comparison_features),
        ("Clasificación de polaridad", classification_metrics),
        ("Clasificación por tipo de lugar", classification_place_type_metrics),
        ("Clustering K-Means", clustering_metrics),
        ("Alineación de clusters con tipo de lugar", clustering_category_alignment),
    ]:
        if table is not None:
            sections.extend([
                html.H4(title, className="mt-3"),
                dbc.Table.from_dataframe(table.round(4), striped=True, bordered=True, hover=True, responsive=True),
            ])
    if tsne_comparison is not None:
        figure = tourist_figure(px.scatter(
            tsne_comparison,
            x="x",
            y="y",
            color="tipo_lugar",
            facet_col="metodo",
            hover_data=["polaridad", "texto"],
            title="Comparación t-SNE de documentos: TF-IDF, Word2Vec y BETO",
            height=520,
            color_discrete_map=PLACE_COLORS,
        ))
        sections.extend([html.H4("Proyección comparativa de embeddings", className="mt-4"), dcc.Graph(figure=figure)])
    return html.Div(sections) if sections else warning(
        "Ejecuta notebooks/05_comparacion_final.ipynb para generar las métricas de accuracy, F1 y silhouette."
    )


@app.callback(Output("tab-content", "children"), Input("tab", "value"))
def render_tab(tab):
    return {"overview": overview_layout, "word2vec": word2vec_layout, "beto": beto_layout, "comparison": comparison_layout}[tab]()


@app.callback(Output("search-results", "children"), Input("search", "n_clicks"), State("query", "value"), prevent_initial_call=True)
def semantic_search(_, query):
    if not query:
        return warning("Escribe una consulta antes de buscar.")
    if beto_embeddings is None:
        return warning("La búsqueda requiere data/models/beto_embeddings.npy. Genera el archivo con el notebook de BETO.")
    if len(beto_embeddings) != len(df):
        return dbc.Alert("Los embeddings y el corpus tienen distinto número de filas. Regenera BETO con el corpus actual.", color="danger")
    try:
        # Se importa aquí para que el resumen del dashboard pueda abrirse aun
        # antes de instalar Transformers/PyTorch o generar BETO.
        from beto_utils import BETOAnalyzer, BETOEmbedder

        embedder = BETOEmbedder()
        results = BETOAnalyzer(embedder, beto_embeddings, df["texto"].tolist()).semantic_search(query, top_k=5)
        output = pd.DataFrame([{"posición": rank, "similitud": round(float(score), 4), "tipo": df.iloc[idx]["tipo_lugar"], "calificación": df.iloc[idx]["calificacion"], "reseña": text[:220]} for rank, (idx, score, text) in enumerate(results, 1)])
        return dbc.Table.from_dataframe(output, striped=True, bordered=True, hover=True, responsive=True)
    except Exception as exc:
        return dbc.Alert(f"No se pudo completar la búsqueda: {exc}", color="danger")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
