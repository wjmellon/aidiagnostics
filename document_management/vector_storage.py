import os
import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_openai.embeddings import OpenAIEmbeddings

def initialize_vectorstore(chunks):
    """Initialize Weaviate vector store and load document chunks."""
    client = weaviate.Client(
        embedded_options=weaviate.embedded.EmbeddedOptions(
            additional_env_vars={
                "ENABLE_MODULES": "text2vec-openai,generative-openai"
            }
        )
    )

    # Define schema if not exists
    if not client.schema.contains({"class": "DocumentChunk"}):
        schema = {
            "class": "DocumentChunk",
            "properties": [
                {"name": "text", "dataType": ["text"], "description": "Chunk of text"},
                {"name": "title", "dataType": ["string"], "description": "Title of the document"},
                {"name": "authors", "dataType": ["string[]"], "description": "Authors of the document"},
                {"name": "index", "dataType": ["int"], "description": "Index of the chunk"},
                {"name": "doi_url", "dataType": ["string"], "description": "DOI link of paper"}
            ],
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "generative-openai": {}
            }
        }
        client.schema.create_class(schema)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        by_text=False,
    )
    return vectorstore.as_retriever()

def initialize_port_retriever(port="8080"):
    """Initialize a retriever from Weaviate vector store at specified port."""
    client = weaviate.Client(
        url=f"http://localhost:{port}",
        additional_headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')}
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Weaviate(
        client=client,
        index_name="DocumentChunk",
        text_key="text",
        attributes=["title", "authors", "doi_url"]
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "score_threshold": 0.8,
            "include_metadata": True,
            "metadata_fields": ["text", "authors", "title", "doi_url"]
        }
    )

    return retriever
