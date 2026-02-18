"""
Model handling classification of available learning resources and creating embeddings to assist with recommendations

"""

import resource
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


class ContentModel:
    """
    A model for generating embeddings for learning resources.
    
    Uses sentence transformers to create semantic embeddings that capture
    the meaning and content of educational resources.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        """
        self.model_name = model_name
        self.embedding_model = None
        self.embeddings_mapping: dict[int, np.ndarray] = {}
        self.data_path = "data/"
        self.text_path = "urls/"
        self.file_path = "files/"
        self.embedding_path = "embeddings/"
        self.resource_name = "resources.csv"
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
    
    def load_model(self):
        """"""
        try:
            
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"Loaded model: {self.model_name}")
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

        if self.embedding_model is None:
            self.load_model()
        
        # Create text representations
        texts = resources_df.apply(
            self.create_text_representation, 
            axis=1
        ).tolist()
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} resources...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
  
        for idx, resource_id in enumerate(resources_df['id']):  
            self.embeddings_mapping[resource_id] = embeddings[idx]
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
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
        Encode input text into an embedding vector.
        """
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_resource_embedding(self, resource: pd.Series) -> np.ndarray:
        """
        Get saved embedding when provided an existing resource.
        """
        resource_id = resource['id']
        embedding = self.get_embedding(resource_id)
        if embedding is None:
            raise ValueError(f"No embedding found for resource ID: {resource_id}")
        return embedding
        

    
    def get_similar_resources(self, embedding: np.ndarray,  resources_df: pd.DataFrame, top_k: int =5) -> list[tuple[int, float, pd.Series]]:
        ##Change to data frames list to return a simple object type.
        ##Try to use simpler object structures and test for performance differences. 
        ## Add option to provide a filtered list of embeddings. Allows for prefiltering based on metadata.
        """
        Find resources similar to user input. 
        params:
            query_text: str : The input text query.
            resources_df: pd.DataFrame : List of resources.
        """
        
 
        query_similarities = self.embedding_model.similarity(embedding, self.embeddings)
        query_similarities_np = query_similarities.cpu().numpy()
        if query_similarities_np.ndim > 1:
            query_similarities_np = query_similarities_np.flatten()
        query_sorted_indices = np.argsort(query_similarities_np)
        query_top_indices = query_sorted_indices[-5:][::-1]
        similar_resources = []
        for idx in query_top_indices:
            similar_resources.append((idx, query_similarities_np[idx], resources_df.iloc[idx]))

        return similar_resources
    



        # Get top 5 most similar embeddings
        # Convert to nparray for reverse argsort
        similarities_np = similarities.cpu().numpy() 
        if similarities_np.ndim > 1:
            similarities_np = similarities_np.flatten()
        
        sorted_indices = np.argsort(similarities_np)
        top_indices = sorted_indices[-5:][::-1]

        # Display corresponding resources

        similar_resources = []
        for idx in query_top_indices:
            similar_resources.append((idx, query_similarities_np[idx], resources_df.iloc[idx]))





# Testing
if __name__ == "__main__":

    
    
    # Create a sample DataFrame
    ##data = pd.read_csv(model.data_path + 'resources.csv').head(10) 
    file_name = "resources.csv"
    model = ContentModel()
    model.load_model()
    data = pd.read_csv(model.data_path + file_name)
    resources_df = pd.DataFrame(data)
    
   
    embeddings = model.generate_embeddings(resources_df)
    model.save_embeddings(embeddings, model.embedding_file_name)