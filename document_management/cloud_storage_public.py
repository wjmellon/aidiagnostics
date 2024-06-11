import os

import pandas as pd
import requests
from bs4 import BeautifulSoup
import weaviate
from weaviate.auth import AuthApiKey
import re
from weaviate.embedded import EmbeddedOptions

def scrape_author_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        author_meta = soup.find('meta', attrs={"name": "author"})
        authors = author_meta['content'] if author_meta else "Unknown Author"

        main_content = soup.find('article')  # Assuming 'article' tag wraps the main content
        text = main_content.get_text(strip=True) if main_content else "No Content Found"

        return authors, text
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return "Unknown Author", "No Content Found"

def chunk_text(source_text, chunk_size=150, overlap_size=25):
    text_words = source_text.split()
    chunks = []
    for i in range(0, len(text_words), chunk_size - overlap_size):
        chunk = " ".join(text_words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Load CSV file containing URLs and Titles
df = pd.read_csv('./sources.csv')
df['URLs'] = df['URL'].apply(lambda x: re.findall(r'https?://\S+', x))
df = df.explode('URLs').dropna(subset=['URLs'])

client = weaviate.Client(
    url=f"http://localhost:8081",  # Dynamically set the port
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

# client = weaviate.Client(embedded_options=EmbeddedOptions(additional_env_vars={
#     "ENABLE_MODULES": "text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai"
# }))

chunk_class = {
    "class": "DocumentChunk",
    "properties": [
        {"name": "text", "dataType": ["text"], "description": "Chunk of text"},
        {"name": "title", "dataType": ["string"], "description": "Title of the document"},
        {"name": "authors", "dataType": ["string"], "description": "Authors of the document"},
        {"name": "index", "dataType": ["int"], "description": "Index of the chunk"},
        {"name": "doi_url", "dataType": ["string"], "description": "DOI link of paper"}
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
    for _, row in df.iterrows():
        url = row['URLs']
        title = row['Title']
        authors, full_text = scrape_author_from_url(url)
        chunks = chunk_text(full_text)
        for i, chunk in enumerate(chunks):
            batch.add_data_object(
                data_object={
                    "title": title,
                    "authors": authors,
                    "text": chunk,
                    "index": i,
                    "doi_url" : url
                },
                class_name="DocumentChunk"
            )

print("All chunks have been uploaded to Weaviate.")
