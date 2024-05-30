from pymed import PubMed
import weaviate
import json
from datetime import datetime
import os


def fetch_from_pubmed(search_term, max_results=500):
    """Fetches top cited and relevant scientific papers related to a query from the PubMed API."""
    pubmed = PubMed(tool="PubMedSearcher", email="myemail@ccc.com")
    # Calculate the year for 15 years ago
    current_year = datetime.now().year
    year_15_years_ago = current_year - 15
    # The date filter is added to the query string.
    query = f"{search_term} AND (cancer[MeSH Terms] OR melanoma[MeSH Terms])"
    results = pubmed.query(query, max_results=max_results)
    articles = []

    for article in results:
        articleDict = article.toDict()
        pub_date = articleDict.get('pub_date', '')
        if pub_date:
            pub_year = int(pub_date[:4])
            if pub_year < year_15_years_ago:
                continue

        authors_list = [author['lastname'] + ', ' + author['forename'] for author in articleDict.get('authors', []) if
                        'lastname' in author and 'forename' in author]
        articles.append({
            'title': articleDict.get('title', 'No title provided'),
            'abstract': articleDict.get('abstract', 'No abstract provided'),
            'authors': authors_list,
            'pubmed_id': articleDict.get('pubmed_id', '').partition('\n')[0],
            'pub_date': pub_date
        })

    # Sort the articles by relevance or any other criteria if needed
    # Here, we assume that the results from PubMed are already sorted by relevance
    return articles


def chunk_text(title, text, chunk_size=150, overlap_size=25):
    """Prepends the title to the text and chunks the combined text into smaller parts with overlapping segments."""
    full_text = f"Title: {title}. {text}"
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def initialize_weaviate():
    """Initializes the Weaviate client and configures the schema."""
    client = weaviate.Client(
        url=f"http://localhost:8080",  # Dynamically set the port
        additional_headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')}
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


def load_into_weaviate(articles, client):
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for article in articles:
            chunks = chunk_text(article['title'], article['abstract'])
            for i, chunk in enumerate(chunks):
                batch.add_data_object(
                    data_object={
                        "title": article['title'],
                        "authors": article['authors'],  # Now passing authors as a list
                        "text": chunk,
                        "index": i,
                        "pubmed_id": article['pubmed_id']
                    },
                    class_name="DocumentChunk"
                )


if __name__ == "__main__":
    search_term = "skin cancer"
    articles = fetch_from_pubmed(search_term)
    weaviate_client = initialize_weaviate()
    load_into_weaviate(articles, weaviate_client)
    print("All chunks have been uploaded to Weaviate.")
