"""
w2v_utils.py
Utilidades para entrenar y analizar modelos Word2Vec.
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle


class Word2VecTrainer:
    """Entrenador de modelos Word2Vec."""
    
    def __init__(self, vector_size=100, window=5, min_count=3, workers=4, seed=42):
        """
        Inicializar parámetros de entrenamiento.
        
        Args:
            vector_size: dimensión de embeddings
            window: tamaño del contexto
            min_count: frecuencia mínima de palabras
            workers: número de threads
            seed: random seed
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.seed = seed
        self.cbow_model = None
        self.skipgram_model = None
    
    def train_cbow(self, tokenized_corpus, epochs=10):
        """
        Entrenar modelo CBOW (Continuous Bag of Words).
        
        Args:
            tokenized_corpus: lista de listas de tokens
            epochs: número de épocas de entrenamiento
            
        Returns:
            modelo entrenado
        """
        print("Entrenando CBOW...")
        self.cbow_model = Word2Vec(
            sentences=tokenized_corpus,
            sg=0,  # 0 = CBOW
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=epochs,
            seed=self.seed,
            negative=10,
            sample=1e-3,
            alpha=0.025,
            min_alpha=0.0001
        )
        print(f"✓ CBOW entrenado. Vocabulario: {len(self.cbow_model.wv)} palabras")
        return self.cbow_model
    
    def train_skipgram(self, tokenized_corpus, epochs=10):
        """
        Entrenar modelo Skip-Gram.
        
        Args:
            tokenized_corpus: lista de listas de tokens
            epochs: número de épocas de entrenamiento
            
        Returns:
            modelo entrenado
        """
        print("Entrenando Skip-Gram...")
        self.skipgram_model = Word2Vec(
            sentences=tokenized_corpus,
            sg=1,  # 1 = Skip-Gram
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=epochs,
            seed=self.seed,
            negative=10,
            sample=1e-3,
            alpha=0.025,
            min_alpha=0.0001
        )
        print(f"✓ Skip-Gram entrenado. Vocabulario: {len(self.skipgram_model.wv)} palabras")
        return self.skipgram_model
    
    def save_model(self, model, filepath):
        """Guardar modelo entrenado."""
        model.save(filepath)
        print(f"✓ Modelo guardado en {filepath}")
    
    def load_model(self, filepath):
        """Cargar modelo entrenado."""
        model = Word2Vec.load(filepath)
        return model


class Word2VecAnalyzer:
    """Análisis de modelos Word2Vec."""
    
    def __init__(self, model):
        """
        Inicializar con un modelo entrenado.
        
        Args:
            model: modelo Word2Vec
        """
        self.model = model
        self.wv = model.wv
    
    def get_vocabulary(self):
        """Obtener vocabulario del modelo."""
        return list(self.wv.index_to_key)
    
    def similarity(self, word1, word2):
        """
        Similitud coseno entre dos palabras.
        
        Args:
            word1, word2: palabras
            
        Returns:
            similitud (0-1)
        """
        try:
            return self.wv.similarity(word1, word2)
        except KeyError:
            return None
    
    def most_similar(self, word, topn=5):
        """
        Palabras más similares a una palabra.
        
        Args:
            word: palabra de referencia
            topn: número de resultados
            
        Returns:
            lista de (palabra, similitud)
        """
        try:
            return self.wv.most_similar(word, topn=topn)
        except KeyError:
            return []
    
    def analogy(self, word_a, word_b, word_c, topn=5):
        """
        Resolver analogía: A es a B como C es a ???
        
        Args:
            word_a, word_b, word_c: palabras
            topn: número de candidatos
            
        Returns:
            lista de (palabra, similitud)
        """
        try:
            results = self.wv.most_similar(positive=[word_b, word_c], negative=[word_a], topn=topn)
            return results
        except KeyError:
            return []
    
    def vocabulary_by_context(self, texts_by_category):
        """
        Analizar vocabulario por categoría/contexto.
        
        Args:
            texts_by_category: dict {categoría: [palabras]}
            
        Returns:
            DataFrame con análisis por categoría
        """
        results = []
        
        for category, words in texts_by_category.items():
            # Palabras únicas en esta categoría
            unique_words = set(words)
            
            # Palabras que sí están en el vocabulario
            vocab_words = [w for w in unique_words if w in self.wv]
            
            results.append({
                'categoría': category,
                'palabras_totales': len(unique_words),
                'en_vocab': len(vocab_words),
                'cobertura': len(vocab_words) / len(unique_words) if unique_words else 0
            })
        
        return pd.DataFrame(results)
    
    def semantic_field(self, seed_word, topn=10):
        """
        Explorar campo semántico alrededor de una palabra.
        
        Args:
            seed_word: palabra de referencia
            topn: número de palabras relacionadas
            
        Returns:
            DataFrame con palabras y similitudes
        """
        similar_words = self.most_similar(seed_word, topn=topn)
        
        if not similar_words:
            return pd.DataFrame()
        
        df = pd.DataFrame(similar_words, columns=['palabra', 'similitud'])
        return df
    
    def similarity_matrix(self, words):
        """
        Matriz de similitud entre un conjunto de palabras.
        
        Args:
            words: lista de palabras
            
        Returns:
            numpy array (matriz de similitud)
        """
        vectors = []
        valid_words = []
        
        for word in words:
            if word in self.wv:
                vectors.append(self.wv[word])
                valid_words.append(word)
        
        if not vectors:
            return None, []
        
        vectors = np.array(vectors)
        sim_matrix = cosine_similarity(vectors)
        
        return sim_matrix, valid_words
    
    def get_vector(self, word):
        """Obtener vector de una palabra."""
        if word in self.wv:
            return self.wv[word]
        return None
    
    def compare_models(self, other_model, words=None):
        """
        Comparar similitudes entre dos modelos.
        
        Args:
            other_model: otro modelo Word2Vec
            words: palabras a comparar (None = top words)
            
        Returns:
            DataFrame con comparación
        """
        if words is None:
            # Usar palabras más frecuentes
            words = self.wv.index_to_key[:100]
        
        results = []
        
        for word in words:
            sim1 = self.similarity(word, word) if word in self.wv else None
            sim2 = other_model.wv.similarity(word, word) if word in other_model.wv else None
            
            # Buscar similares en cada modelo
            similar1 = self.most_similar(word, topn=3)
            similar2 = other_model.wv.most_similar(word, topn=3) if word in other_model.wv else []
            
            results.append({
                'palabra': word,
                'top_similares_modelo1': [w[0] for w in similar1],
                'top_similares_modelo2': [w[0] for w in similar2]
            })
        
        return pd.DataFrame(results)


class Word2VecVisualizer:
    """Visualizaciones para Word2Vec."""
    
    def __init__(self, analyzer):
        """
        Inicializar con un analizador.
        
        Args:
            analyzer: Word2VecAnalyzer
        """
        self.analyzer = analyzer
    
    def prepare_tsne_data(self, words=None, topn=200):
        """
        Preparar datos para visualización t-SNE.
        
        Args:
            words: palabras específicas (None = más frecuentes)
            topn: número de palabras más frecuentes
            
        Returns:
            (vectores, palabras)
        """
        if words is None:
            words = self.analyzer.wv.index_to_key[:topn]
        
        vectors = []
        valid_words = []
        
        for word in words:
            if word in self.analyzer.wv:
                vectors.append(self.analyzer.wv[word])
                valid_words.append(word)
        
        return np.array(vectors), valid_words
    
    def get_similarity_data(self, words):
        """
        Datos para heatmap de similitud.
        
        Args:
            words: lista de palabras
            
        Returns:
            (matriz de similitud, palabras válidas)
        """
        return self.analyzer.similarity_matrix(words)


def save_analyzer(analyzer, filepath):
    """Guardar analizador."""
    with open(filepath, 'wb') as f:
        pickle.dump(analyzer, f)
    print(f"✓ Analizador guardado en {filepath}")


def load_analyzer(filepath):
    """Cargar analizador."""
    with open(filepath, 'rb') as f:
        analyzer = pickle.load(f)
    return analyzer
