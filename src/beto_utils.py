"""
beto_utils.py
Utilidades para cargar y usar BETO (Spanish BERT).
"""

import re

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class BETOEmbedder:
    """Generador de embeddings con BETO."""
    
    def __init__(self, model_name="dccuchile/bert-base-spanish-wwm-cased", device=None):
        """
        Inicializar BETO.
        
        Args:
            model_name: nombre del modelo en HuggingFace
            device: 'cuda' o 'cpu' (auto-detectar si None)
        """
        print(f"Cargando {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Detectar dispositivo
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ BETO cargado en {self.device}")
        print(f"  Parámetros: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Dimensión: {self.model.config.hidden_size}")
    
    def get_sentence_embedding(self, text, use_cls=True):
        """
        Obtener embedding de una oración.
        
        Args:
            text: texto
            use_cls: usar token [CLS] (True) o promedio (False)
            
        Returns:
            vector numpy (768,)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if use_cls:
            # Usar token [CLS] (posición 0)
            embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        else:
            # Usar promedio de todos los tokens (excepto [CLS] y [SEP])
            last_hidden = outputs.last_hidden_state[0, 1:-1, :]
            embedding = last_hidden.mean(dim=0).cpu().numpy()
        
        return embedding
    
    def get_token_embeddings(self, text):
        """
        Obtener embeddings de tokens individuales.
        
        Args:
            text: texto
            
        Returns:
            dict {token: vector}
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        embeddings = outputs.last_hidden_state[0].cpu().numpy()
        
        return {token: embeddings[i] for i, token in enumerate(tokens)}

    def get_word_embedding(self, text, word):
        """Devuelve el embedding contextual de una palabra dentro de un texto.

        Si el tokenizador divide la palabra en subpalabras, se promedian sus
        vectores. Esto evita confundir un embedding de oración ([CLS]) con la
        representación contextual de la palabra que se desea analizar.
        """
        if not isinstance(text, str) or not isinstance(word, str):
            return None

        match = re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)
        if match is None:
            return None

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = self.model(**model_inputs).last_hidden_state[0].cpu().numpy()

        start, end = match.span()
        positions = [
            index for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        if not positions:
            return None
        return hidden[positions].mean(axis=0)
    
    def embed_batch(self, texts, use_cls=True, batch_size=32):
        """
        Generar embeddings para múltiples textos.
        
        Args:
            texts: lista de textos
            use_cls: usar [CLS] o promedio
            batch_size: tamaño de batch
            
        Returns:
            numpy array (n_texts, 768)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generando embeddings"):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            if use_cls:
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                batch_embeddings = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


class BETOAnalyzer:
    """Análisis semántico con BETO."""
    
    def __init__(self, embedder, embeddings=None, texts=None):
        """
        Inicializar analizador.
        
        Args:
            embedder: BETOEmbedder
            embeddings: matriz de embeddings precalculados
            texts: textos correspondientes a los embeddings
        """
        self.embedder = embedder
        self.embeddings = embeddings
        self.texts = texts
    
    def similarity(self, text1, text2):
        """
        Similitud entre dos textos.
        
        Args:
            text1, text2: textos
            
        Returns:
            similitud (0-1)
        """
        emb1 = self.embedder.get_sentence_embedding(text1)
        emb2 = self.embedder.get_sentence_embedding(text2)
        
        sim = cosine_similarity([emb1], [emb2])[0, 0]
        return sim
    
    def semantic_search(self, query, top_k=5):
        """
        Búsqueda semántica: encontrar textos más similares a query.
        
        Args:
            query: texto de búsqueda
            top_k: número de resultados
            
        Returns:
            lista de (índice, similitud, texto)
        """
        if self.embeddings is None or self.texts is None:
            raise ValueError("Embeddings y textos necesarios para búsqueda")
        
        query_emb = self.embedder.get_sentence_embedding(query)
        
        # Similitud con todos los embeddings
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        
        # Top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (idx, similarities[idx], self.texts[idx])
            for idx in top_indices
        ]
        
        return results
    
    def analyze_polysemy(self, word, contexts):
        """
        Analizar polisemia: vectores diferentes para la misma palabra en contextos.
        
        Args:
            word: palabra a analizar
            contexts: lista de oraciones con la palabra
            
        Returns:
            DataFrame con embeddings y similitudes
        """
        embeddings = []
        
        for context in contexts:
            token_embeds = self.embedder.get_token_embeddings(context)
            
            # Buscar la palabra en los tokens
            for token, emb in token_embeds.items():
                if word.lower() in token.lower():
                    embeddings.append({
                        'contexto': context,
                        'embedding': emb,
                        'token': token
                    })
                    break
        
        if not embeddings:
            return pd.DataFrame()
        
        # Calcular similitudes entre embeddings
        vectors = np.array([e['embedding'] for e in embeddings])
        sim_matrix = cosine_similarity(vectors)
        
        df = pd.DataFrame(embeddings)
        df['num_contexto'] = range(len(embeddings))
        
        return df, sim_matrix
    
    def clustering(self, n_clusters=5):
        """
        Clustering de textos.
        
        Args:
            n_clusters: número de clusters
            
        Returns:
            array de labels
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        return labels, kmeans
    
    def get_cluster_summary(self, labels, top_n=3):
        """
        Resumen de clusters.
        
        Args:
            labels: labels de clusters
            top_n: número de textos representativos por cluster
            
        Returns:
            DataFrame con información de clusters
        """
        summaries = []
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_texts = [self.texts[i] for i in range(len(labels)) if mask[i]]
            cluster_embeddings = self.embeddings[mask]
            
            # Centroide del cluster
            centroid = cluster_embeddings.mean(axis=0)
            
            # Textos más cercanos al centroide
            distances = cosine_similarity([centroid], cluster_embeddings)[0]
            top_indices = np.argsort(distances)[-top_n:][::-1]
            top_texts = [cluster_texts[i] for i in top_indices]
            
            summaries.append({
                'cluster': cluster_id,
                'tamaño': len(cluster_texts),
                'textos_representativos': ' | '.join([t[:100] + '...' for t in top_texts])
            })
        
        return pd.DataFrame(summaries)


class BETOLanguageModel:
    """BETO como modelo de lenguaje (Masked LM)."""
    
    def __init__(self, model_name="dccuchile/bert-base-spanish-wwm-cased"):
        """Inicializar pipeline de fill-mask."""
        print("Cargando pipeline de Masked LM...")
        self.pipe = pipeline(
            "fill-mask",
            model=model_name
        )
        print("✓ Pipeline cargado")
    
    def predict_mask(self, text, top_k=5):
        """
        Predecir palabra tapada ([MASK]).
        
        Args:
            text: texto con [MASK]
            top_k: número de predicciones
            
        Returns:
            lista de (palabra, probabilidad)
        """
        if '[MASK]' not in text:
            raise ValueError("El texto debe contener [MASK]")
        
        results = self.pipe(text, top_k=top_k)
        
        return [(r['token_str'].strip(), r['score']) for r in results]
    
    def analyze_context_predictions(self, contexts_with_masks):
        """
        Analizar predicciones en múltiples contextos.
        
        Args:
            contexts_with_masks: lista de textos con [MASK]
            
        Returns:
            DataFrame con análisis
        """
        results = []
        
        for context in contexts_with_masks:
            predictions = self.predict_mask(context, top_k=3)
            
            results.append({
                'contexto': context,
                'predicciones': predictions,
                'palabra_mas_probable': predictions[0][0] if predictions else None,
                'confianza': predictions[0][1] if predictions else 0
            })
        
        return pd.DataFrame(results)


class BETOVisualizer:
    """Visualizaciones para BETO."""
    
    def __init__(self, analyzer):
        """
        Inicializar con un analizador.
        
        Args:
            analyzer: BETOAnalyzer
        """
        self.analyzer = analyzer
    
    def prepare_tsne_data(self, sample_size=None, random_state=42):
        """
        Preparar datos para visualización t-SNE.
        
        Args:
            sample_size: número de muestras (None = todas)
            random_state: seed
            
        Returns:
            (embeddings, textos)
        """
        if sample_size is None or sample_size >= len(self.analyzer.embeddings):
            return self.analyzer.embeddings, self.analyzer.texts
        
        np.random.seed(random_state)
        indices = np.random.choice(
            len(self.analyzer.embeddings),
            size=sample_size,
            replace=False
        )
        
        embeddings = self.analyzer.embeddings[indices]
        texts = [self.analyzer.texts[i] for i in indices]
        
        return embeddings, texts


def save_embeddings(embeddings, filepath):
    """Guardar embeddings a archivo."""
    np.save(filepath, embeddings)
    print(f"✓ Embeddings guardados en {filepath}")


def load_embeddings(filepath):
    """Cargar embeddings desde archivo."""
    return np.load(filepath)
