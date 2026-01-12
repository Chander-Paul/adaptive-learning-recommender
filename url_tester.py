"""
A module to test URLs and suggest alternatives using Google Custom Search API."""

import requests
import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def check_url(url: str) -> bool:
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False
    


def google_rest_search(query: str, api_key: str, search_engine_id: str, num_results: int = 10) -> list[dict]:
    
    response = requests.get(f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}:omuauf_lfve&q={query}&callback=hndlr&num={num_results}", timeout=5)
    if response.status_code == 200:
        results = response.json()
        search_results = []
        if 'items' in results:
            for item in results['items']:
                search_results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
        return search_results
    else:
        print(f"Error: Received status code {response.status_code}")
        return []

def google_api_search(query: str, api_key: str, search_engine_id: str, num_results: int = 10) -> list[dict]:
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        
        result = service.cse().list(
            q=query,
            cx=search_engine_id,
            num=num_results
        ).execute()

        search_results = []
        if 'items' in result:
            for item in result['items']:
                search_results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
        return search_results
    
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def suggest_alternative_url(title, topic_name, author):

    try:
        with open('key.yaml', 'r') as file:
            config = yaml.safe_load(file)
            api_key = config.get('google_api_key')
            search_engine_id = config.get('search_engine_id')
        
        if not api_key or not search_engine_id:
            print("Error: Missing API key or Search Engine ID in key.yaml")
            return []
        

        query = f"{title} {topic_name} {author}"
        results = google_custom_search(query, api_key, search_engine_id)
        
        return results
    
    except FileNotFoundError:
        print("Error: key.yaml file not found")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    # Test URL checking
    url = "https://research.facebook.com/research/babi/"
    print(f"Testing URL: {url}")
    
    if check_url(url):
        print(f"✓ {url} is working")
    else:
        print(f"✗ {url} is not accessible")
        print("\nSearching for alternatives...")
        
        results = suggest_alternative_url("The bAbi Project", "Machine Learning", "Facebook")
        
        if results:
            print(f"\nFound {len(results)} alternative results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   URL: {result['link']}")
                print(f"   {result['snippet']}")
        else:
            print("No alternative results found.")
