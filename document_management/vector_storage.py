import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_openai.embeddings import OpenAIEmbeddings
from weaviate.embedded import EmbeddedOptions
from document_management.document_chunker import load_and_chunk_document
import os
from langchain.retrievers.multi_vector import MultiVectorRetriever

from langchain.storage import InMemoryByteStore


def initialize_vectorstore(chunks):
    # Set up the Weaviate client
    client = weaviate.Client(embedded_options=EmbeddedOptions(additional_env_vars={
        "ENABLE_MODULES": "text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai"
    }))

    # Create the schema
    schema = {
        "classes": [
            {
                "class": "Paper",
                "description": "Represents an academic paper",
                "properties": [
                    {"name": "title", "dataType": ["string"], "description": "The title of the paper"},
                    {"name": "authors", "dataType": ["string[]"], "description": "A list of authors of the paper"}
                ]
            },
            {
                "class": "Author",
                "description": "Represents an author of academic papers",
                "properties": [
                    {"name": "name", "dataType": ["string"], "description": "The name of the author"},
                    {"name": "papers", "dataType": ["Paper"], "description": "Papers written by the author", "cardinality": "many"}
                ]
            }
        ]
    }


    client.schema.delete_all()

    # Create new schema
    client.schema.create(schema)

    # Customize OpenAIEmbeddings instantiation
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",  # Specify the model name
        dimensions=3072  # Specify the dimension, only if the OpenAIEmbeddings class allows setting this parameter
    )

    # Initialize the vector store
    vectorstore = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        by_text=False,
    )
    return vectorstore.as_retriever()


#test_query = client.query.get('Chunk', properties=['text', 'index', 'document']).do()
#print(test_query)

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever

import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_openai.embeddings import OpenAIEmbeddings
import os

def initialize_port_retriever(port="8080"):
    # Initialize Weaviate client connecting to the Docker instance
    client = weaviate.Client(
        url=f"http://localhost:{port}",
        additional_headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')}
    )

    # Specify the model to use for embedding (e.g., 'text2vec-openai' or 'multi2vec-clip')
    # This could be dynamic or based on a config
    embedding_model = os.getenv('EMBEDDING_MODEL', 'text2vec-openai')  # Default to 'text2vec-openai'

    # Customize embedding model based on environment variables or configuration
    if embedding_model == 'text2vec-openai':
        # OpenAI embeddings model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"  # Specify the OpenAI embedding model
        )
    elif embedding_model == 'multi2vec-clip':
        # Use the CLIP model specified in the docker-compose.yml
        embeddings = OpenAIEmbeddings(
            model="multi2vec-clip"  # Assuming the Weaviate client handles this model
        )
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")

    # Weaviate vector store setup for the existing "DocumentChunk" collection
    vectorstore = Weaviate(
        client=client,
        index_name="DocumentChunk",  # Use the existing collection name
        text_key="text",  # Specify the field used for the text data
        attributes=["title", "authors", "doi_url"]  # Attributes to retrieve as metadata
    )

    # Initialize retriever with search parameters
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,  # Number of results to return
            "score_threshold": 0.8,  # Relevance score threshold
            "include_metadata": True,  # Include metadata in the results
            "metadata_fields": ["text", "authors", "title", "doi_url"]  # Metadata fields to include
        }
    )

    return retriever



