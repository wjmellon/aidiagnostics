import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_openai.embeddings import OpenAIEmbeddings
from weaviate.embedded import EmbeddedOptions
from document_management.document_chunker import load_and_chunk_document
import os

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

def initialize_cloud_retriever():
    client = weaviate.Client(
        url="http://localhost:8080",  # Ensure this is your correct cloud instance URL
        additional_headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')}
    )

    # Initialize the vector store with the correct index and text key
    vectorstore = Weaviate(
        client=client,
        index_name="DocumentChunk",  # Replace with your actual class name in Weaviate
        text_key="text"  # Replace with the property name that contains the main text data
    )

    return vectorstore.as_retriever()



