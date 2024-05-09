import pandas as pd
import requests
from bs4 import BeautifulSoup
import weaviate
from weaviate.auth import AuthApiKey
import re
import os

def scrape_details_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('title').text if soup.find('title') else "No Title Found"
        author_meta = soup.find('meta', attrs={"name": "author"})
        authors = author_meta['content'] if author_meta else "Unknown Author"

        main_content = soup.find('article')  # Assuming 'article' tag wraps the main content
        text = main_content.get_text(strip=True) if main_content else "No Content Found"

        return title, authors, text
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return "No Title Found", "Unknown Author", "No Content Found"

def chunk_text(source_text, chunk_size=150, overlap_size=25):
    text_words = source_text.split()
    chunks = []
    for i in range(0, len(text_words), chunk_size - overlap_size):
        chunk = " ".join(text_words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Load CSV file containing URLs
df = pd.read_csv('./citations.csv')
df['URLs'] = df['Citation'].apply(lambda x: re.findall(r'https?://\S+', x))
urls = df['URLs'].explode().dropna().unique().tolist()

# Initialize Weaviate client
client = weaviate.Client(
    url="https://aidiag-yrjn4mqf.weaviate.network",  # Ensure this is your correct cloud instance URL
    auth_client_secret=weaviate.auth.AuthApiKey(api_key="PJ2ajygrzY7UICcaUseQxZEhl5K4NReGrLoX"),
    additional_headers={ "X-OpenAI-Api-Key":OPENAI_APIKEY   # <-- Replace with your API key
    }
    )


chunk_class = {
    "class": "DocumentChunk",
    "properties": [
        {"name": "text", "dataType": ["text"], "description": "Chunk of text"},
        {"name": "title", "dataType": ["string"], "description": "Title of the document"},
        {"name": "authors", "dataType": ["string"], "description": "Authors of the document"},
        {"name": "index", "dataType": ["int"], "description": "Index of the chunk"}
    ],
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "generative-openai": {}
    }
}
client.schema.delete_all()
client.schema.create_class(chunk_class)

# Scrape, chunk, and upload to Weaviate
client.batch.configure(batch_size=100)
with client.batch as batch:
    for url in urls:
        title, authors, full_text = scrape_details_from_url(url)
        chunks = chunk_text(full_text)
        for i, chunk in enumerate(chunks):
            batch.add_data_object(
                data_object={
                    "title": title,
                    "authors": authors,
                    "text": chunk,
                    "index": i
                },
                class_name="DocumentChunk"
            )

print("All chunks have been uploaded to Weaviate.")
