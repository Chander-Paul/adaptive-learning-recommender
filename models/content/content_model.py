"""
Model handling classification of available learning resources and creating embeddings to assist with recommendations

"""

import resource
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Literal

from helper_functions import map_resource_medium_to_type


class ContentModel:
    """
    A model for generating embeddings for learning resources.
    
    Uses sentence transformers bi-encoders to create semantic embeddings that capture
    the meaning and content of educational resources.
    
    bi-encoder embeddings used for large scale similarity searches and recommendations. -> faster
    cross-encoder scores can be used for smaller datasets or more precise relevance scoring. -> slower but more accurate for relevance

    Look into using cross-encoder embeddins for topic modelling training.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 encoding_type: Literal['bi-encoder', 'cross-encoder'] = 'bi-encoder',
                 cross_encoder_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the ContentModel.
        
        Args:
            model_name: Name of the bi-encoder model (used when encoding_type='bi-encoder')
            encoding_type: Type of encoding to use ('bi-encoder' or 'cross-encoder')
            cross_encoder_model_name: Name of the cross-encoder model (used when encoding_type='cross-encoder')
        """
        self.model_name = model_name
        self.encoding_type = encoding_type
        self.cross_encoder_model_name = cross_encoder_model_name
        self.embedding_model = None
        self.cross_encoder_model = None
        self.embeddings_mapping: dict[int, np.ndarray] = {}
        self.data_path = "data/"
        self.text_path = "urls/"
        self.file_path = "files/"
        self.embedding_path = "embeddings/"
        self.resource_name = "resources.csv"
        
        # Set embedding file names based on encoding type
        if encoding_type == 'cross-encoder':
            self.embedding_file_name = self.cross_encoder_model_name.replace('/', '_') + "_" + self.resource_name + "_scores.npy"
            self.embedding_mapping_name = self.cross_encoder_model_name.replace('/', '_') + "_" + self.resource_name + "_scores_mapping.npy"
        else:
            self.embedding_file_name = self.model_name+"_"+self.resource_name+"_embeddings.npy"
            self.embedding_mapping_name = self.model_name+"_"+self.resource_name+"_embeddings_mapping.npy"
        
        self.embeddings = None

    
    def load_resources(self, file_name: str) -> pd.DataFrame:
        """
        Load resources from a CSV file into a DataFrame.
        """
        data = pd.read_csv(self.data_path + file_name)
        resources_df = pd.DataFrame(data)
        return resources_df
    
    def load_model(self, model_type: str | None = None) -> None:
        """Load either the bi-encoder or cross-encoder model.

        If ``model_type`` is omitted, default to this instance's ``encoding_type``.
        """
        model_to_load = model_type or self.encoding_type

        try:
            if model_to_load == 'cross-encoder':
                if self.cross_encoder_model is None:
                    self.cross_encoder_model = CrossEncoder(self.cross_encoder_model_name)
                    print(f"Loaded cross-encoder model: {self.cross_encoder_model_name}")
            else:
                if self.embedding_model is None:
                    self.embedding_model = SentenceTransformer(self.model_name)
                    print(f"Loaded bi-encoder model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def create_text_representation(self, resource: pd.Series, ignore_empty: bool = False, 
                                   ignore_headers: bool = False,
                                   ignore_title: bool = False,
                                   ignore_author: bool = False,
                                   ignore_topic: bool = False
                                   ) -> str:

        parts = []
        ##May need to remove  title and author fields to prevent overfitting.
        ## Noticing "Autthor" in many embeddings.
        ## Remove headers from text to prevent overfitting 
        
        if pd.notna(resource.get('title')) and not ignore_title:
            parts.append(f"{resource['title']}")
        
        if pd.notna(resource.get('topic_name')) and not ignore_topic:
            parts.append(f"{resource['topic_name']}")
        
        if pd.notna(resource.get('author')) and not ignore_author:
            parts.append(f"{resource['author']}")

        

        if pd.notna(resource.get('txtpath')):
            try:
                f = open(self.text_path + resource['txtpath'], 'r', encoding='utf-8')
                text_content = f.read()
                parts.append(f"{ self.preprocess_text(text_content)}")
            except FileNotFoundError:
                if not ignore_empty:
                    parts.append("")
                else:
                    print(f"Warning: Text file not found for resource ID {resource['id']} at path {self.text_path + resource['txtpath']}. Skipping text content.")
                    parts = []

  
        
        return ". ".join(parts)
    
    def generate_embeddings(self, 
                          resources_df: pd.DataFrame,
                          batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for resources using the configured bi-encoder  
 
        
        Args:
            resources_df: DataFrame containing resources
            batch_size: Batch size for encoding
            """

        if self.embedding_model is None:
            self.load_model()
        
        # Create text representations
        texts = resources_df.apply(
            self.create_text_representation, 
            axis=1
        ).tolist()
        
        # Generate embeddings
        print(f"Generating bi-encoder embeddings for {len(texts)} resources...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        for idx, resource_id in enumerate(resources_df['id']):  
            self.embeddings_mapping[resource_id] = embeddings[idx]
        
        print(f"Generated bi-encoder embeddings with shape: {embeddings.shape}")
        return embeddings
    
    
    def get_embedding(self, resource_id: int) -> np.ndarray | None:
        return self.embeddings_mapping.get(resource_id)
    




    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            
            reader = PdfReader(file_path)
            text_parts = []
            reader.flat
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n".join(text_parts)
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
        


    def save_embeddings(self, embeddings: np.ndarray, file_path: str) -> None:
        
        if not os.path.exists(self.embedding_path):
            os.makedirs(self.embedding_path)
        
        np.save(self.embedding_path + file_path, embeddings)
        np.save(self.embedding_path + file_path.replace('.npy', '_mapping.npy'), 
            np.array(list(self.embeddings_mapping.items()), dtype=object))
        print(f"Saved embeddings to {self.embedding_path + file_path}")

    def load_embeddings(self, file_path: str) -> np.ndarray:
        self.embeddings = np.load(self.embedding_path + file_path)
        mapping_array = np.load(self.embedding_path + file_path.replace('.npy', '_mapping.npy'), allow_pickle=True)
        self.embeddings_mapping = dict(mapping_array.tolist())
        print(f"Loaded embeddings from {self.embedding_path + file_path} with shape: {self.embeddings.shape}")
        return self.embeddings


    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text to remove html tags and extra whitespace. Heavy preprocessing is avoided for sbert enconding.
    
        """

        text = BeautifulSoup(text, 'lxml').get_text()
    
        text = ' '.join(text.split())
        
        return text
    
    def calculate_simple_document_complexity(self, text: str) -> float:
        """
        Calculate the Flesch-Kincaid readability score of the text.(possibly add more complexity calculations later)
        A higher score indicates easier readability.
        """


        # Flesch-Kincaid Reading Ease formula

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        sentences = sent_tokenize(text)
        sentences = [s for s in sentences if s.strip()]
        num_sentences = len(sentences)
        words = re.findall(r'\w+', text)
        num_words = len(words)
        num_syllables = sum(self.count_syllables(word) for word in words)
        
        if num_sentences == 0 or num_words == 0:
            return 0.0
        
        
        fk_score = 206.835 - (1.015 * (num_words / num_sentences))  - (84.6 * (num_syllables / num_words))        
        
        return fk_score
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode input text into an embedding or score vector. Bi-emcpdomg for fast user queries
        
        """

        
        if self.embedding_model is None:
            self.load_model()
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def score_text_pair(self, text1: str, text2: str) -> float:
        """
        Score the relevance/similarity between two texts using cross-encoder.

        """
        
        if self.cross_encoder_model is None:
            self.load_model(model_type='cross-encoder')
        if self.cross_encoder_model is None:
            raise RuntimeError("Cross-encoder model not loaded.")     
        else:  
            score = self.cross_encoder_model.predict([text1, text2])
        return float(score)
    
    def get_resource_embedding(self, resource: pd.Series) -> np.ndarray | int:
        """
        Get embedding or index for a resource.
        
        For bi-encoders: returns the embedding vector
        """
        resource_id = resource['id']
        embedding = self.get_embedding(resource_id)
        if embedding is None:
            raise ValueError(f"No embedding found for resource ID: {resource_id}")
        return embedding
        

    
    def get_similar_resources(self, embedding: np.ndarray,  resources_df: pd.DataFrame, top_k: int =5) -> list[tuple[int, float, pd.Series]]:
        """
        Find resources similar to given embedding.
        
        For bi-encoders: uses cosine similarity on fixed embeddings
        For cross-encoders: uses the precomputed cross-encoder scores
        
        params:
            embedding: np.ndarray or int - embedding vector (bi-encoder) or row index (cross-encoder)
            resources_df: pd.DataFrame - list of resources
            top_k: int - number of top similar resources to return
        """
        query_similarities = self.embedding_model.similarity(embedding, self.embeddings)
        query_similarities_np = query_similarities.cpu().numpy()
        if query_similarities_np.ndim > 1:
            query_similarities_np = query_similarities_np.flatten()
        query_sorted_indices = np.argsort(query_similarities_np)
        query_top_indices = query_sorted_indices[-top_k:][::-1]
        similar_resources = []
        for idx in query_top_indices:
            similar_resources.append((idx, query_similarities_np[idx], resources_df.iloc[idx]))

        return similar_resources
    
    
    def score_query_against_resources(self, query_text: str, resources_df: pd.DataFrame, 
                                     top_k: int = 5) -> list[tuple[int, float, pd.Series]]:
        """
        Score a query text against all resources using cross-encoder.
        
        Only available when encoding_type='cross-encoder'.
        Useful for search/query scenarios.
        
        Args:
            query_text: Query text to score against
            resources_df: DataFrame of resources
            top_k: Number of top similar resources to return
            
        Returns:
            List of tuples (index, score, resource_series)
        """
        if self.encoding_type != 'cross-encoder':
            raise RuntimeError(
                "score_query_against_resources() is only available for cross-encoders. "
                "Use get_similar_resources() for bi-encoders."
            )
        
        if self.cross_encoder_model is None:
            self.load_model()
        
        # Create text representations
        texts = resources_df.apply(
            self.create_text_representation, 
            axis=1
        ).tolist()
        
        # Create query-candidate pairs
        pairs = [[query_text, text] for text in texts]
        
        # Get cross-encoder scores
        scores = self.cross_encoder_model.predict(pairs, show_progress_bar=False)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((idx, float(scores[idx]), resources_df.iloc[idx]))
        
        return results

    def map_medium_to_type(self, medium: str) -> str:
        """
        Map a learner's preferred medium to a resource type for filtering.
        This is a simple mapping and can be expanded based on the dataset and use case.
        """
        mapped_type = map_resource_medium_to_type(medium)
        if mapped_type is None:
            raise ValueError(f"Unsupported medium: {medium}")
        return mapped_type
    



# Testing
if __name__ == "__main__":
    
    file_name = "resources.csv"
    
    # Example 1: Using bi-encoder (default)
    print("=" * 60)
    print("Example 1: Bi-encoder embeddings")
    print("=" * 60)
    model_bi = ContentModel()
    model_bi.load_model()
    data = pd.read_csv(model_bi.data_path + file_name)
    resources_df = pd.DataFrame(data)
    
    embeddings = model_bi.generate_embeddings(resources_df)
    model_bi.save_embeddings(embeddings, model_bi.embedding_file_name)
    
    print("\n")
    
    # Example 2: Using cross-encoder
    print("=" * 60)
    print("Example 2: Cross-encoder scores")
    print("=" * 60)
    model_ce = ContentModel(
        encoding_type='cross-encoder',
        cross_encoder_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'
    )
    model_ce.load_model()
    
