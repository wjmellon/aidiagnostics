import os
import logging
import requests
import io
import pdfplumber
import weaviate
from bs4 import BeautifulSoup
from pdfminer.pdfparser import PDFSyntaxError
import time
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_doi_url(doi):
    """Ensure that the DOI is formatted as a full URL."""
    if not doi.startswith('http://') and not doi.startswith('https://'):
        doi_url = f"https://doi.org/{doi}"
    else:
        doi_url = doi
    return doi_url

def query_unpaywall(query_terms, email):
    """Query the Unpaywall API for multiple search terms and collect DOIs."""
    logging.info(f"Querying Unpaywall for terms: {query_terms}")
    dois = []
    for query_term in query_terms:
        formatted_query = query_term.replace(" ", "%20")
        url = f"https://api.unpaywall.org/v2/search/?query={formatted_query}&email={email}&is_oa=true"
        logging.info(f"Querying URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        term_dois = [result['response']['doi'] for result in results['results']]
        logging.info(f"Found DOIs for term '{query_term}': {term_dois}")
        dois.extend(term_dois)
    return dois

def get_document_details(doi, email):
    """Retrieve document details including PDF URL, title, and authors."""
    doi_url = ensure_doi_url(doi)
    url = f"https://api.unpaywall.org/v2/{doi_url}?email={email}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    logging.info(f"Fetching document details for DOI: {doi_url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        pdf_url = data.get('best_oa_location', {}).get('url_for_pdf', '')
        landing_page_url = data.get('best_oa_location', {}).get('url_for_landing_page', '')
        title = data.get('title', 'Unknown Title')

        z_authors = data.get('z_authors') or []
        authors = [f"{author.get('given', '')} {author.get('family', '')}".strip() for author in z_authors]

        logging.info(f"Document details - Title: {title}, Authors: {authors}, PDF URL: {pdf_url}, Landing Page URL: {landing_page_url}")
        return (pdf_url if pdf_url else landing_page_url, title, authors)

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred while fetching document details: {e}")
        return None, 'Unknown Title', []
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None, 'Unknown Title', []

def download_and_extract_text(url):
    """Downloads a PDF from a given URL and extracts text or scrapes text if not a PDF."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    logging.info(f"Downloading and extracting text from URL: {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' in content_type:
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = ''.join(page.extract_text() or '' for page in pdf.pages)
                logging.info(f"Extracted text from PDF (first 100 chars): {text[:100]}...")
                return text
        else:
            # If not a PDF, assume HTML or a text content type and scrape text
            text = scrape_and_clean_text(response.content)
            logging.info(f"Extracted text from HTML (first 100 chars): {text[:100]}...")
            return text
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e}")
        return None
    except PDFSyntaxError:
        logging.error("Not a valid PDF file.")
        return scrape_and_clean_text(url)  # Try scraping as HTML as a fallback
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

def scrape_and_clean_text(content):
    """Scrape text from HTML content."""
    logging.info(f"Scraping and cleaning text from content.")
    try:
        soup = BeautifulSoup(content, 'html.parser')
        for script in soup(["script", "style", "header", "footer", "nav", "form"]):
            script.decompose()
        cleaned_text = ' '.join(soup.stripped_strings)
        logging.info(f"Cleaned text (first 100 chars): {cleaned_text[:100]}...")
        return cleaned_text
    except Exception as e:
        logging.error(f"Failed to scrape or clean HTML: {e}")
        return None

def chunk_text(source_text, max_tokens=6000, chunk_size=200, overlap_size=50):
    """Splits the text into smaller chunks ensuring each chunk stays within the token limit."""
    logging.info(f"Chunking text into smaller pieces.")
    encoding = tiktoken.get_encoding("cl100k_base")  # Use the encoding used by the OpenAI model

    words = source_text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        token_count = len(encoding.encode(chunk))

        while token_count > max_tokens and end > start:
            end -= 10  # Reduce the chunk size incrementally
            chunk = " ".join(words[start:end])
            token_count = len(encoding.encode(chunk))

        if token_count > 0:  # Ensure we are adding valid chunks
            chunks.append(chunk)
            logging.info(f"Chunk size: {token_count} tokens")  # Monitor token sizes

        start = end - overlap_size  # Move to the next chunk with overlap

    return chunks

def initialize_weaviate():
    """Initializes the Weaviate client and configures the schema."""
    logging.info(f"Initializing Weaviate client.")
    client = weaviate.Client(
        url="http://localhost:8080",
        additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
    )

    if not client.schema.exists("DocumentChunk"):
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
        client.schema.create_class(chunk_class)
        logging.info(f"Created schema for 'DocumentChunk' class.")
    else:
        logging.info(f"'DocumentChunk' class already exists in the schema.")

    return client

def upload_chunks_to_weaviate(client, text_chunks, title, authors, doi_url):
    """Uploads text chunks to Weaviate."""
    logging.info(f"Uploading chunks to Weaviate.")
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
                logging.warning(f"Batch ReadTimeout Exception occurred! Retrying in {retry_delay * attempt}s. [{attempt}/{retries}]")
                time.sleep(retry_delay * attempt)
            else:
                logging.error("Batch ReadTimeout Exception occurred! No more retries left.")
                raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during upload: {e}")
            raise

def main():
    email = os.getenv('UNPAYWALL_EMAIL', 'your_email@example.com')
    if email == 'your_email@example.com':
        logging.error("Please set your 'UNPAYWALL_EMAIL' in the environment variables.")
        return

    # Define your list of query terms
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
        logging.info(f"Processing DOI: {doi}")
        url, title, authors = get_document_details(doi, email)
        if url:
            full_text = download_and_extract_text(url)
            if full_text:
                text_chunks = chunk_text(full_text)
                upload_chunks_to_weaviate(weaviate_client, text_chunks, title, authors, url)
            else:
                logging.warning(f"No text available for DOI: {doi}")
        else:
            logging.warning(f"No URL found for DOI: {doi}")

if __name__ == "__main__":
    main()
