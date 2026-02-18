"""Quick tests for embeddings generation and retrieval."""

import pytest
import numpy as np
import pandas as pd
from content_model import ContentModel
import os


if __name__ == "__main__":
  
    

    file_name = "resources.csv"
    model = ContentModel()
    model.load_model()
    
    data = pd.read_csv(model.data_path + file_name)
    resources_df = pd.DataFrame(data)
    embeddings = model.load_embeddings(model.embedding_file_name)
    print("Resources DataFrame:")
    
    #Test against Random Existing Resource
    def test_random_resource():
        random_resource = resources_df.sample(n=1).index[0]
        print(resources_df.loc[random_resource])
        # Quick test using 1st/ Random Resource

        
        # Calculate cosine similarity with all embeddings
        first_embedding = embeddings[random_resource]
        similarities = model.embedding_model.similarity(first_embedding, embeddings)



        # Get top 5 most similar embeddings
        # Convert to nparray for reverse argsort
        similarities_np = similarities.cpu().numpy() 
        if similarities_np.ndim > 1:
            similarities_np = similarities_np.flatten()
        
        sorted_indices = np.argsort(similarities_np)
        print(f"Type after argsort: {type(sorted_indices)}")
        top_indices = sorted_indices[-5:][::-1]

        print(f"Top 5 most similar indices: {top_indices}")
        print(f"Similarities: {similarities_np[top_indices]}")

        # Display corresponding resources
        print("\nTop 5 most similar resources:")
        for idx in top_indices:
            print(f"\nIndex {idx} (Similarity: {similarities_np[idx]:.4f}):")
            print(resources_df.iloc[idx])


    # Test against new query
    query_text = "Introduction to machine learning and data science."
    query_embedding = model.embedding_model.encode(query_text, convert_to_numpy=True)
    query_similarities = model.embedding_model.similarity(query_embedding, embeddings)
    query_similarities_np = query_similarities.cpu().numpy()
    if query_similarities_np.ndim > 1:
        query_similarities_np = query_similarities_np.flatten()
    query_sorted_indices = np.argsort(query_similarities_np)
    query_top_indices = query_sorted_indices[-5:][::-1]
    print(f"\nTop 5 resources similar to the query: {query_text}")
    for idx in query_top_indices:
        print(f"\nIndex {idx} (Similarity: {query_similarities_np[idx]:.4f}):")
        print(resources_df.iloc[idx])


  