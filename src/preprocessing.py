"""
preprocessing.py
Funciones de preprocesamiento para análisis de reseñas turísticas.
"""

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from tqdm import tqdm

# Descargar recursos necesarios
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Preprocesador de texto para reseñas en español."""
    
    def __init__(self, language='spanish', remove_stopwords=True, use_lemma=False):
        """
        Inicializar preprocesador.
        
        Args:
            language: idioma para stopwords
            remove_stopwords: si eliminar palabras comunes
            use_lemma: si usar lematización (requiere spaCy)
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.use_lemma = use_lemma
        
        # Stopwords en español
        self.stop_words = set(stopwords.words('spanish'))
        self.stop_words.update(['tour', 'tours', 'excursión', 'excursiones', 'lugar', 'lugares'])
        
        # Cargar modelo de spaCy si se necesita lematización
        if use_lemma:
            try:
                self.nlp = spacy.load("es_core_news_sm")
            except OSError:
                print("Descargando modelo de spaCy...")
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', 'es_core_news_sm'], quiet=True)
                self.nlp = spacy.load("es_core_news_sm")
        else:
            self.nlp = None
    
    def clean_text(self, text):
        """
        Limpiar texto: remover HTML, URLs, caracteres especiales.
        
        Args:
            text: texto a limpiar
            
        Returns:
            texto limpio
        """
        if not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remover emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remover menciones y hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remover caracteres especiales pero mantener acentos
        text = re.sub(r'[^a-záéíóúñ\s]', '', text)
        
        # Remover espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenizar texto en palabras.
        
        Args:
            text: texto a tokenizar
            
        Returns:
            lista de tokens
        """
        text = self.clean_text(text)
        tokens = word_tokenize(text, language=self.language)
        return tokens
    
    def lemmatize(self, text):
        """
        Lematizar texto usando spaCy.
        
        Args:
            text: texto a lematizar
            
        Returns:
            texto lematizado
        """
        if not self.use_lemma or self.nlp is None:
            return text
        
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc]
        return ' '.join(lemmas)
    
    def process(self, text, remove_stopwords=None):
        """
        Procesar texto completo: limpiar, tokenizar, remover stopwords.
        
        Args:
            text: texto a procesar
            remove_stopwords: override del atributo de la clase
            
        Returns:
            lista de tokens procesados
        """
        if remove_stopwords is None:
            remove_stopwords = self.remove_stopwords
        
        # Limpiar
        text = self.clean_text(text)
        
        # Lematizar si aplica
        if self.use_lemma:
            text = self.lemmatize(text)
        
        # Tokenizar
        tokens = self.tokenize(text)
        
        # Remover stopwords y tokens cortos
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        else:
            tokens = [t for t in tokens if len(t) > 2]
        
        return tokens
    
    def process_dataframe(self, df, text_column='texto', progress_bar=True):
        """
        Procesar columna de texto en DataFrame.
        
        Args:
            df: DataFrame con textos
            text_column: nombre de la columna de texto
            progress_bar: mostrar barra de progreso
            
        Returns:
            DataFrame con nueva columna 'tokens'
        """
        df = df.copy()
        
        iterator = tqdm(df[text_column], desc="Procesando textos") if progress_bar else df[text_column]
        df['tokens'] = [self.process(text) for text in iterator]
        
        return df


def load_corpus(filepath):
    """
    Cargar corpus desde archivo CSV.
    
    Args:
        filepath: ruta al archivo CSV
        
    Returns:
        DataFrame con el corpus
    """
    df = pd.read_csv(filepath)
    print(f"Corpus cargado: {len(df)} reseñas")
    print(f"Columnas: {list(df.columns)}")
    print(f"Tipos de lugar: {df['tipo_lugar'].unique()}")
    print(f"Rango de calificaciones: {df['calificacion'].min()}-{df['calificacion'].max()}")
    return df


def split_by_category(df, category_column, text_column='texto'):
    """
    Dividir corpus por categoría.
    
    Args:
        df: DataFrame
        category_column: columna para dividir
        text_column: columna de texto
        
    Returns:
        dict: {categoría: lista de textos}
    """
    return {cat: df[df[category_column] == cat][text_column].tolist() 
            for cat in df[category_column].unique()}


def get_tokenized_corpus(df, preprocessor=None, text_column='texto'):
    """
    Obtener corpus tokenizado para entrenar embeddings.
    
    Args:
        df: DataFrame
        preprocessor: TextPreprocessor (crear uno si None)
        text_column: columna de texto
        
    Returns:
        lista de listas de tokens
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor(remove_stopwords=True)
    
    df_processed = preprocessor.process_dataframe(df, text_column=text_column)
    return df_processed['tokens'].tolist()


def sample_corpus(df, n_samples=None, seed=42):
    """
    Muestrear corpus (útil para testing rápido).
    
    Args:
        df: DataFrame
        n_samples: número de muestras (None = usar todo)
        seed: random seed
        
    Returns:
        DataFrame muestreado
    """
    if n_samples is None or n_samples >= len(df):
        return df
    return df.sample(n=n_samples, random_state=seed)
