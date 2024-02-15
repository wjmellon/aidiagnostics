

from data_processing.data_fetcher import download_document
from document_management.document_chunker import load_and_chunk_document
from document_management.vector_storage import initialize_vectorstore
from rag_components.generator_setup import setup_prompt, build_rag_chain
from query_handling.query_processor import handle_query

# Disable SSL warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Assuming you have set your OpenAI API key in the environment variables
import os
print(os.environ['OPENAI_API_KEY'])

# Define your constants and configurations
URL = "https://raw.githubusercontent.com/wjmellon/aidiagnostics/main/testlink1.txt"
PATH_TO_SAVE = "data/aggregated.txt"
TEMPLATE_STR = """You are an assistant for question-answering tasks. These questions are about Acral Lentiginous Melanoma.
You must use the provided pieces of context to answer questions. 
If you don't know the answer, just say that you don't know. 
Answer in a clinical dermatology setting. Give the user a citation from the text, author, section, and quote from text.
Question: {question} 
Context: {context} 
Answer:
"""

# Use the modular functions
download_document(URL, PATH_TO_SAVE)
chunks = load_and_chunk_document(PATH_TO_SAVE)
retriever = initialize_vectorstore(chunks)
prompt = setup_prompt(TEMPLATE_STR)
rag_chain = build_rag_chain(retriever, prompt)

# Start the query handling process
handle_query(rag_chain)
