"""
dashboard_utils.py
Utilidades para preparar datos del dashboard.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class DashboardDataPreparator:
    """Preparación de datos para visualizaciones del dashboard."""
    
    @staticmethod
    def prepare_tsne(embeddings, labels=None, sample_size=None, perplexity=30, seed=42):
        """
        Preparar proyección t-SNE.
        
        Args:
            embeddings: matriz de embeddings
            labels: etiquetas para colorear
            sample_size: número de puntos (None = todas)
            perplexity: parámetro de t-SNE
            seed: random seed
            
        Returns:
            (coord_2d, labels_out)
        """
        if sample_size and sample_size < len(embeddings):
            np.random.seed(seed)
            indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
            embeddings_sample = embeddings[indices]
            if labels is not None:
                labels_sample = [labels[i] for i in indices]
            else:
                labels_sample = None
        else:
            embeddings_sample = embeddings
            labels_sample = labels
        
        # Ajustar perplexity
        max_perplexity = (len(embeddings_sample) - 1) / 3
        perplexity = min(perplexity, max_perplexity)
        
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, n_iter=1000)
        coords_2d = tsne.fit_transform(embeddings_sample)
        
        return coords_2d, labels_sample
    
    @staticmethod
    def create_scatter_3d(coords_2d, labels=None, texts=None, title="t-SNE Projection"):
        """
        Crear gráfico scatter interactivo.
        
        Args:
            coords_2d: coordenadas 2D
            labels: etiquetas para color
            texts: hover texts
            title: título del gráfico
            
        Returns:
            figura Plotly
        """
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'label': labels if labels is not None else [''] * len(coords_2d),
            'text': texts if texts is not None else [f"Punto {i}" for i in range(len(coords_2d))]
        })
        
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='label' if labels is not None else None,
            hover_data={'text': True, 'x': ':.2f', 'y': ':.2f'},
            title=title,
            labels={'x': 't-SNE Dim 1', 'y': 't-SNE Dim 2'},
            height=600
        )
        
        fig.update_layout(
            hovermode='closest',
            showlegend=True,
            font=dict(size=11)
        )
        
        return fig
    
    @staticmethod
    def create_similarity_heatmap(similarity_matrix, labels=None, title="Similarity Matrix"):
        """
        Crear heatmap de similitud.
        
        Args:
            similarity_matrix: matriz de similitud
            labels: etiquetas de filas/columnas
            title: título
            
        Returns:
            figura Plotly
        """
        if labels is None:
            labels = [f"Item {i}" for i in range(len(similarity_matrix))]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='YlOrRd',
            text=np.round(similarity_matrix, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='%{y} - %{x}: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=700,
            font=dict(size=11)
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(data, x, y, title="Bar Chart", color=None):
        """
        Crear gráfico de barras.
        
        Args:
            data: DataFrame
            x: columna para eje X
            y: columna para eje Y
            title: título
            color: columna para color
            
        Returns:
            figura Plotly
        """
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            height=400
        )
        
        fig.update_layout(
            hovermode='x unified',
            font=dict(size=11)
        )
        
        return fig
    
    @staticmethod
    def create_rating_distribution(df, color_column=None):
        """
        Crear distribución de calificaciones.
        
        Args:
            df: DataFrame con columna 'calificacion'
            color_column: columna para agrupar colores
            
        Returns:
            figura Plotly
        """
        rating_counts = df['calificacion'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(x=rating_counts.index, y=rating_counts.values, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="Distribución de Calificaciones",
            xaxis_title="Calificación",
            yaxis_title="Cantidad",
            height=400,
            showlegend=False,
            font=dict(size=11)
        )
        
        return fig
    
    @staticmethod
    def create_place_type_distribution(df):
        """
        Crear distribución por tipo de lugar.
        
        Args:
            df: DataFrame con columna 'tipo_lugar'
            
        Returns:
            figura Plotly
        """
        place_counts = df['tipo_lugar'].value_counts()
        
        fig = px.pie(
            values=place_counts.values,
            names=place_counts.index,
            title="Distribución por Tipo de Lugar",
            height=450
        )
        
        fig.update_layout(font=dict(size=11))
        
        return fig
    
    @staticmethod
    def create_rating_by_place(df):
        """
        Crear gráfico de calificación promedio por tipo de lugar.
        
        Args:
            df: DataFrame
            
        Returns:
            figura Plotly
        """
        rating_by_place = df.groupby('tipo_lugar')['calificacion'].agg(['mean', 'count']).reset_index()
        rating_by_place.columns = ['tipo_lugar', 'calificacion_promedio', 'cantidad']
        
        fig = px.bar(
            rating_by_place,
            x='tipo_lugar',
            y='calificacion_promedio',
            color='cantidad',
            title="Calificación Promedio por Tipo de Lugar",
            labels={'calificacion_promedio': 'Calificación Promedio', 'cantidad': 'Cantidad'},
            height=400
        )
        
        fig.update_layout(hovermode='x unified', font=dict(size=11))
        
        return fig
    
    @staticmethod
    def create_metrics_comparison(metrics_dict, title="Comparison of Metrics"):
        """
        Crear tabla comparativa de métricas.
        
        Args:
            metrics_dict: dict con métodos como keys y dicts de métricas como values
            title: título
            
        Returns:
            figura Plotly (tabla)
        """
        data = []
        for method, metrics in metrics_dict.items():
            row = {'Method': method}
            row.update(metrics)
            data.append(row)
        
        df_metrics = pd.DataFrame(data)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_metrics.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[df_metrics[col] for col in df_metrics.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(title=title, height=300)
        
        return fig


class SearchResultsFormatter:
    """Formatear resultados de búsqueda semántica."""
    
    @staticmethod
    def format_search_results(results, df=None):
        """
        Formatear resultados de búsqueda.
        
        Args:
            results: lista de (índice, similitud, texto)
            df: DataFrame original (opcional, para agregar metadatos)
            
        Returns:
            DataFrame formateado
        """
        formatted = []
        
        for idx, (result_idx, similarity, text) in enumerate(results, 1):
            row = {
                'ranking': idx,
                'similitud': f"{similarity:.4f}",
                'texto': text[:200] + '...' if len(text) > 200 else text
            }
            
            if df is not None and result_idx < len(df):
                row['calificacion'] = df.iloc[result_idx]['calificacion']
                row['tipo_lugar'] = df.iloc[result_idx]['tipo_lugar']
            
            formatted.append(row)
        
        return pd.DataFrame(formatted)
    
    @staticmethod
    def create_search_results_table(results_df):
        """
        Crear tabla Plotly de resultados.
        
        Args:
            results_df: DataFrame con resultados
            
        Returns:
            figura Plotly
        """
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(results_df.columns),
                fill_color='steelblue',
                align='left',
                font=dict(color='white', size=11)
            ),
            cells=dict(
                values=[results_df[col] for col in results_df.columns],
                fill_color='aliceblue',
                align='left',
                font=dict(size=10),
                height=30
            )
        )])
        
        fig.update_layout(height=400)
        
        return fig


class StatisticsCalculator:
    """Calcular estadísticas del corpus."""
    
    @staticmethod
    def corpus_statistics(df):
        """
        Calcular estadísticas generales del corpus.
        
        Args:
            df: DataFrame
            
        Returns:
            dict con estadísticas
        """
        return {
            'total_resenas': len(df),
            'calificacion_promedio': df['calificacion'].mean(),
            'calificacion_min': df['calificacion'].min(),
            'calificacion_max': df['calificacion'].max(),
            'tipos_lugar': df['tipo_lugar'].nunique(),
            'fuentes': df['fuente'].nunique() if 'fuente' in df.columns else 0
        }
    
    @staticmethod
    def embedding_statistics(embeddings):
        """
        Calcular estadísticas de embeddings.
        
        Args:
            embeddings: matriz de embeddings
            
        Returns:
            dict con estadísticas
        """
        return {
            'total_embeddings': len(embeddings),
            'dimensiones': embeddings.shape[1],
            'norma_promedio': np.linalg.norm(embeddings, axis=1).mean(),
            'norma_min': np.linalg.norm(embeddings, axis=1).min(),
            'norma_max': np.linalg.norm(embeddings, axis=1).max()
        }


def create_overview_layout(corpus_stats, embedding_stats):
    """
    Crear layout con estadísticas generales.
    
    Args:
        corpus_stats: dict de estadísticas del corpus
        embedding_stats: dict de estadísticas de embeddings
        
    Returns:
        lista de dcc.Card para mostrar
    """
    cards = []
    
    # Cards del corpus
    for key, value in corpus_stats.items():
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        else:
            display_value = str(value)
        
        cards.append({
            'title': key.replace('_', ' ').title(),
            'value': display_value
        })
    
    # Cards de embeddings
    for key, value in embedding_stats.items():
        if isinstance(value, float):
            display_value = f"{value:.4f}"
        else:
            display_value = str(value)
        
        cards.append({
            'title': key.replace('_', ' ').title(),
            'value': display_value
        })
    
    return cards
