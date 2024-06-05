import os

import requests
import io
import pdfplumber
import weaviate
from bs4 import BeautifulSoup
from pdfminer.pdfparser import PDFSyntaxError

def ensure_doi_url(doi):
    """Ensure that the DOI is formatted as a full URL."""
    return f"https://doi.org/{doi}" if not doi.startswith('http://') and not doi.startswith('https://') else doi

def query_unpaywall(query_terms, email):
    """Query the Unpaywall API for multiple search terms and collect DOIs."""
    dois = []
    for query_term in query_terms:
        url = f"https://api.unpaywall.org/v2/search/?query={query_term}&email={email}&is_oa=true"
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        term_dois = [result['response']['doi'] for result in results['results']]
        dois.extend(term_dois)
    return dois

def get_pdf_url(doi, email):
    """Retrieve the PDF or landing page URL for a DOI from Unpaywall API."""
    doi_url = ensure_doi_url(doi)
    url = f"https://api.unpaywall.org/v2/{doi_url}?email={email}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    pdf_url = data.get('best_oa_location', {}).get('url_for_pdf')
    landing_page_url = data.get('best_oa_location', {}).get('url_for_landing_page')
    return pdf_url if pdf_url else landing_page_url

def download_and_extract_text(url):
    """Downloads a PDF from a given URL and extracts text or scrapes text if not a PDF."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' in content_type:
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                return ''.join(page.extract_text() or '' for page in pdf.pages)
        else:
            # If not a PDF, assume HTML or a text content type and scrape text
            return scrape_and_clean_text(response.content)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None
    except PDFSyntaxError:
        print("Not a valid PDF file.")
        return scrape_and_clean_text(url)  # Try scraping as HTML as a fallback
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def scrape_and_clean_text(content):
    """Scrape text from HTML content or a URL, removing irrelevant HTML tags."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
    }
    # Check if the content is a URL, assuming it's a string and starts with http:// or https://
    if isinstance(content, str) and (content.startswith("http://") or content.startswith("https://")):
        response = requests.get(content, headers=headers)
        response.raise_for_status()
        html_content = response.content
    else:
        # Assume it's raw HTML content in bytes and needs to be decoded to a string
        html_content = content if isinstance(content, str) else content.decode('utf-8')
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style", "header", "footer", "nav", "form"]):
            script.decompose()
        cleaned_text = ' '.join(soup.stripped_strings)
        return cleaned_text
    except Exception as e:
        print(f"Failed to scrape or clean HTML: {e}")
        return None

def chunk_text(source_text, chunk_size=200, overlap_size=30):
    """Splits the text into smaller chunks."""
    words = source_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def initialize_weaviate():
    """Initializes the Weaviate client and configures the schema."""
    client = weaviate.Client(url="http://localhost:8080", additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")})
    client.schema.delete_all()
    chunk_class = {
        "class": "DocumentChunk",
        "properties": [
            {"name": "text", "dataType": ["text"], "description": "Chunk of text"},
            {"name": "title", "dataType": ["string"], "description": "Title of the document"},
            {"name": "authors", "dataType": ["text[]"], "description": "Authors of the document"},
            {"name": "index", "dataType": ["int"], "description": "Index of the chunk"},
            {"name": "doi_url", "dataType": ["string"], "description": "DOI link of paper"}
        ],
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "generative-openai": {}
        }
    }
    client.schema.create_class(chunk_class)
    return client

def upload_chunks_to_weaviate(client, text_chunks, title, authors, doi_url):
    """Uploads text chunks to Weaviate."""
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for index, chunk in enumerate(text_chunks):
            batch.add_data_object(
                data_object={
                    "title": title,
                    "authors": authors,
                    "text": chunk,
                    "index": index,
                    "doi_url": doi_url
                },
                class_name="DocumentChunk"
            )
        batch.flush()

if __name__ == "__main__":
    email = 'wmellon@asu.edu'  # Replace with your actual email
    query_terms = ["skin cancer", "melanoma", "skin cancer case study", "rare skin cancers", "skin cancer demographics", "carcinoma", "top skin cancers", "skin cancer prognosis"]
    weaviate_client = initialize_weaviate()
    dois = query_unpaywall(query_terms, email)
    for doi in dois:
        doi_url = ensure_doi_url(doi)
        url = get_pdf_url(doi, email)
        full_text = download_and_extract_text(url)
        if full_text:
            text_chunks = chunk_text(full_text)
            upload_chunks_to_weaviate(weaviate_client, text_chunks, "Extracted from PDF or Webpage", ["Unknown"], doi_url)
        else:
            print(f"No text available for DOI: {doi}")
