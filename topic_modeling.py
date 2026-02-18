"""
Context topic modeling for resources using BERTopic
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings

import yaml

import content_model
warnings.filterwarnings('ignore')
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS



class BERTopicModeler:
    """
    
    """
    
    def __init__(self, 
                 n_topics: Optional[int] = None,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 language: str = 'english',
                 verbose: bool = True,
                 custom_stop_words: Optional[List[str]] = None):
        """
        Initialize BERTopic modeler.
        
        Args:
            n_topics: Number of topics (None for automatic detection)
            embedding_model: Name of the sentence transformer model
            language: Language for stop words
            verbose: Whether to print progress
            custom_stop_words: Additional words to ignore (e.g., ['author', 'page'])
        """        
        self.n_topics = n_topics
        self.embedding_model_name = embedding_model
        self.language = language
        self.verbose = verbose
        self.custom_stop_words = custom_stop_words or []
        self.model = None
        self.embeddings = None
    

        
    def create_model(self, min_topic_size: int = 10, nr_topics: Optional[int] = None):
        """
        Args:
            min_topic_size: Minimum number of documents per topic
            nr_topics: Number of topics to reduce to (None for automatic)
        """
        # Embedding model
        embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Vectorizer for topic representation
        # Combine default stop words with custom ones
        if self.custom_stop_words:
            stop_words = list(ENGLISH_STOP_WORDS.union(set(word.lower() for word in self.custom_stop_words)))
        else:
            stop_words = self.language
            
        vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Create BERTopic model
        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics,
            verbose=self.verbose,
            calculate_probabilities=True
        )
        
        return self.model
    
    
    def display_topics(self, n_topics: int = 10, n_words: int = 10):
        """
        Display topics with their top words.
        
        Args:
            n_topics: Number of topics to display
            n_words: Number of words per topic
        """
        topic_info = self.model.get_topic_info()
        
        print(f"\n{'='*80}")
        print(f"Top {n_words} words per topic (BERTopic)")
        print(f"{'='*80}\n")
        
        for idx, row in topic_info.head(n_topics).iterrows():
            if row['Topic'] == -1:
                print(f"Topic {row['Topic']} (Outliers): {row['Count']} documents")
            else:
                print(f"Topic {row['Topic']}: {row['Count']} documents")
                topic_words = self.model.get_topic(row['Topic'], n_words)['Main']
                print(topic_words)
                words_str = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_words])
                print(f"  {words_str}\n")
    
    def visualize_topics(self, save_path: Optional[str] = None):
        """
        Create interactive visualization of topics.
        
        Args:
            save_path: Path to save the HTML visualization
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        fig = self.model.visualize_topics()
        
        if save_path:
            fig.write_html(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            fig.show()
    
    def visualize_hierarchy(self, save_path: Optional[str] = None):
        """
        Create hierarchical visualization of topics.
        
        Args:
            save_path: Path to save the HTML visualization
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        hierarchical_topics = self.model.hierarchical_topics(self.documents)
        fig = self.model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Hierarchy visualization saved to {save_path}")
        else:
            fig.show()
    




def main():
    """
    Main function demonstrating BERTopic workflow.
    """
    from content_model import ContentModel
    
    # setup
    LOAD_MODEL = True
    data_path = 'data/resources.csv'
    use_existing_embeddings = True  # Set to True to use embeddings from ContentModel
 
    print("Initialise Berttopic Moderl for topic modelling.")
    print("="*50)

    
    # Load data
    content_model = ContentModel()
    resources_df = pd.read_csv(data_path)
    print(f"Loaded {len(resources_df)} documents")
    documents = resources_df.apply(content_model.create_text_representation, axis=1).tolist()
    
    #create or load model and load embeddings
    modeler = BERTopicModeler(
        embedding_model='all-MiniLM-L6-v2',##Switch to using a config file to kee track of models.
        verbose=True
    )
    if LOAD_MODEL:
        modeler.model = BERTopic.load('models/bertopic_model')
    else:
        modeler.create_model(
            min_topic_size=15,
            nr_topics=20  # Set to None for automatic topic detection
        )

    embeddings = None
    if use_existing_embeddings:
        try:
            embeddings = content_model.load_embeddings(content_model.embedding_file_name)
            print(f"Loaded existing embeddings with shape: {embeddings.shape}")
        except FileNotFoundError:
            print("Existing embeddings not found. Will compute new ones.")
            print("Run content_model.py first to generate embeddings.")
    
    # Fit and transform 
    topics, probs = modeler.model.fit_transform(documents, embeddings=embeddings)
    
    # Assign topic to resources
    multi_topics = False
    if multi_topics:

        threshold = 0.6  # Assign topics with >60% probability
        multi_topics = []
        for doc_probs in probs:
            doc_topics = [topic_id for topic_id, prob in enumerate(doc_probs) if prob >= threshold]
            multi_topics.append(doc_topics)

        resources_df['all_topics'] = multi_topics
    else:
        resources_df['predicted_topic'] = topics
        resources_df['predicted_topic_prob'] = np.max(probs, axis=1)

    
    # Display topics
    modeler.display_topics(n_topics=15, n_words=10)
    

    topic_info = modeler.model.get_topic_info()
    print("\nTopic Distribution:")
    print(topic_info.head(15).to_string(index=False))
    
    # Sample documents per topic
    print("\n" + "="*80)
    print("SAMPLE DOCUMENTS PER TOPIC")
    print("="*80)
    
    for topic_id in range(0, min(5, len(topic_info))):
        print(f"\nTopic {topic_id}:")
        topic_docs = resources_df[resources_df['predicted_topic'] == topic_id].head(5)
        for idx, row in topic_docs.iterrows():
            print(f"  - {row['title'][:80]}... (prob: {row['predicted_topic_prob']:.3f})")
    
    # Save model
    import os
    os.makedirs('models', exist_ok=True)
    modeler.model.save('models/bertopic_model')
    
    # Save resources with predicted topics
    output_path = 'data/resources_with_bertopic.csv'
    resources_df.to_csv(output_path, index=False)
    print(f"\nEnriched dataset saved to {output_path}")

    # Save Topic Info
    topic_info.to_csv('data/bertopic_topic_info.csv', index=False)
    print("Topic information saved to data/bertopic_topic_info.csv")
    
    # Create visualizations
    try:
        print("\nGenerating visualizations...")
        modeler.visualize_topics(save_path='visualizations/bertopic_topics.html')
        print("Visualizations created!")
    except Exception as e:
        print(f"Could not create visualizations: {e}")


if __name__ == "__main__":
    main()
