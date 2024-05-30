import os
import weaviate
import requests
import pdfplumber
from unpywall import Unpywall
import io

# Set up Unpaywall credentials
os.environ['UNPAYWALL_EMAIL'] = 'wmellon@asu.edu'  # Replace with your email

from pdfminer.pdfparser import PDFSyntaxError


def download_and_extract_text(url):
    """Downloads a PDF from a given URL and extracts text from it. Handles errors more robustly."""
    if not url:
        print("No URL provided for download.")
        return None

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        # Use a context manager to ensure the file is handled properly
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            full_text = ''
            for page in pdf.pages:
                full_text += page.extract_text() or ''
            return full_text
    except requests.RequestException as e:
        print(f"Failed to download the PDF: {e}")
        return None
    except PDFSyntaxError:
        print("Not a valid PDF file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def search_unpaywall(query_term):
    """Searches articles by text query using Unpaywall and extracts required DOIs."""
    results = Unpywall.query(query=query_term, is_oa=True)
    dois = results['doi'].tolist()
    return dois

def initialize_weaviate():
    """Initializes the Weaviate client and configures the schema."""
    client = weaviate.Client(
        url=f"http://localhost:8080",
        additional_headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')}  # Replace with your actual API key
    )
    try:
        client.schema.delete_all()
        chunk_class = {
            "class": "DocumentChunk",
            "properties": [
                {"name": "text", "dataType": ["text"], "description": "Chunk of text"},
                {"name": "title", "dataType": ["string"], "description": "Title of the document"},
                {"name": "authors", "dataType": ["string[]"], "description": "Authors of the document"},
                {"name": "index", "dataType": ["int"], "description": "Index of the chunk"}
            ],
            "vectorizer": "text2vec-openai"
        }
        client.schema.create_class(chunk_class)
    except Exception as e:
        print(f"Failed to manage schema in Weaviate: {e}")
    return client

def chunk_text(text, chunk_size=150, overlap_size=25):
    """Chunks text into smaller parts with overlapping segments."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_into_weaviate(dois, client):
    """Loads detailed article data into Weaviate from a list of DOIs."""
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for doi in dois:
            article_data = Unpywall.get_json(doi)
            title = article_data.get('title', 'No title provided')
            authors = [author['family'] for author in article_data.get('authors', []) if 'family' in author]
            pdf_link = Unpywall.get_pdf_link(doi)
            full_text = download_and_extract_text(pdf_link)
            if not full_text:
                print(f"No text available for DOI: {doi}")
                continue  # Skip this entry if no text could be extracted
            text_chunks = chunk_text(full_text)
            for index, chunk in enumerate(text_chunks):
                batch.add_data_object(
                    data_object={
                        "title": title,
                        "authors": authors,
                        "text": chunk,
                        "index": index
                    },
                    class_name="DocumentChunk"
                )


if __name__ == "__main__":
    query_term = "skin cancer"
    dois = search_unpaywall(query_term)
    weaviate_client = initialize_weaviate()
    load_into_weaviate(dois, weaviate_client)
    print("All articles have been uploaded to Weaviate.")
