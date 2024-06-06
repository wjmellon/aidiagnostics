import os
import logging
import requests
import io
import pdfplumber
import weaviate
from bs4 import BeautifulSoup
from pdfminer.pdfparser import PDFSyntaxError
import time

def ensure_doi_url(doi):
    """Ensure that the DOI is formatted as a full URL."""
    return f"https://doi.org/{doi}" if not doi.startswith('http://') and not doi.startswith('https://') else doi

def query_unpaywall(query_terms, email):
    """Query the Unpaywall API for multiple search terms and collect DOIs."""
    dois = []
    for query_term in query_terms:
        formatted_query = query_term.replace(" ", "%20")
        url = f"https://api.unpaywall.org/v2/search/?query={formatted_query}&email={email}&is_oa=true"
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        term_dois = [result['response']['doi'] for result in results['results']]
        dois.extend(term_dois)
    return dois
def get_document_details(doi, email):
    """Retrieve document details including PDF URL, title, and authors."""
    doi_url = ensure_doi_url(doi)
    url = f"https://api.unpaywall.org/v2/{doi_url}?email={email}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    pdf_url = data.get('best_oa_location', {}).get('url_for_pdf', '')
    landing_page_url = data.get('best_oa_location', {}).get('url_for_landing_page', '')
    title = data.get('title', 'Unknown Title')

    # Correctly handle z_authors being None
    z_authors = data.get('z_authors') or []
    authors = [f"{author.get('given', '')} {author.get('family', '')}".strip() for author in z_authors if author.get('given') or author.get('family')]

    return (pdf_url if pdf_url else landing_page_url, title, authors)


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


import tiktoken


def chunk_text(source_text, max_tokens=6000, chunk_size=200, overlap_size=50):
    """Splits the text into smaller chunks ensuring each chunk stays within the token limit."""
    encoding = tiktoken.get_encoding("cl100k_base")  # Use the encoding used by the OpenAI model

    words = source_text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        token_count = len(encoding.encode(chunk))

        while token_count > max_tokens:
            end -= 10  # Reduce the chunk size incrementally
            chunk = " ".join(words[start:end])
            token_count = len(encoding.encode(chunk))

        chunks.append(chunk)
        start = end - overlap_size  # Move to the next chunk with overlap

    return chunks


def initialize_weaviate():
    """Initializes the Weaviate client and configures the schema."""
    client = weaviate.Client(url="http://localhost:8080", additional_headers={"X-OpenAI-Api-Key": os.getenv(("OPENAI_API_KEY"))})
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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_chunks_to_weaviate(client, text_chunks, title, authors, doi_url):
    """Uploads text chunks to Weaviate."""
    batch_size = 20
    retries = 3
    retry_delay = 2

    client.batch.configure(batch_size=batch_size)

    for attempt in range(1, retries + 1):
        try:
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
            logging.info(f"Successfully uploaded {len(text_chunks)} chunks for DOI {doi_url}")
            break
        except requests.exceptions.ReadTimeout:
            if attempt < retries:
                logging.warning(f"[ERROR] Batch ReadTimeout Exception occurred! Retrying in {retry_delay * attempt}s. [{attempt}/{retries}]")
                time.sleep(retry_delay * attempt)
            else:
                logging.error("[ERROR] Batch ReadTimeout Exception occurred! No more retries left.")
                raise

if __name__ == "__main__":
    email = 'wmellon@asu.edu'  # Replace with your actual email
    query_terms = [
        "skin cancer", "melanoma", "skin cancer case study", "rare skin cancers",
        "skin cancer demographics", "carcinoma", "top skin cancers", "skin cancer prognosis",
        "skin cancer by race", "skin cancer treatment", "clinical skin oncology",
        "clinical oncology", "UV radiation and skin cancer", "skin cancer diagnosis",
        "basal cell carcinoma", "pediatric skin cancer", "skin cancer clinical prevention",
        "skin cancer case reports", "skin cancer epidemiology", "personalized skin cancer therapy",
        "skin cancer metastasis", "skin cancer risk factors", "Basal Cell Carcinoma (BCC)",
        "Squamous Cell Carcinoma (SCC)", "Melanoma", "Merkel Cell Carcinoma",
        "Cutaneous Lymphoma", "Kaposi Sarcoma", "Actinic Keratosis",
        "Dermatofibrosarcoma Protuberans (DFSP)", "Sebaceous Carcinoma", "Atypical Fibroxanthoma"
    ]
    weaviate_client = initialize_weaviate()
    dois = query_unpaywall(query_terms, email)
    for doi in dois:
        url, title, authors = get_document_details(doi, email)
        full_text = download_and_extract_text(url)
        if full_text:
            text_chunks = chunk_text(full_text)
            upload_chunks_to_weaviate(weaviate_client, text_chunks, title, authors, url)
        else:
            print(f"No text available for DOI: {doi}")
