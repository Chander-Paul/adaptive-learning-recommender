"""
Script for downloading documents from TutorialBank URLs for use in the creation of learning resource embeddings.
"""

import requests
import os
import pandas as pd


def download_resources(urls: list[tuple[str,str]], download_dir: str) -> None:
    """Download documents from the given URLs and save them to the specified directory."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    for url, name in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            filename = os.path.join(download_dir, name.replace(" ", "_") + os.path.splitext(url)[-1])
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {url} to {filename}")
        
        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")

# Testing
if __name__ == "__main__":

    # Load the TSV file
    df = pd.read_csv('data/resources-v2023-clean.tsv', sep='\t')


    resources = df[['url', 'title', 'topic_name']].head(10)
    print(resources)


    download_directory = "resources"
    download_resources(resources['url','name'].tolist(), download_directory)