import ssl
import os
from data_processing.data_fetcher import download_document
from document_management.document_chunker import load_and_chunk_document
from document_management.vector_storage import initialize_vectorstore
from rag_components.generator_setup import setup_prompt, initialize_llms, query_llms


ssl._create_default_https_context = ssl._create_unverified_context

URL = "https://github.com/wjmellon/aidiagnostics/blob/main/data/aggregated.txt"
PATH_TO_SAVE = "data/aggregated.txt"
TEMPLATE_STR = """You are an assistant for question-answering tasks. These questions are about skin cancer.
You must use the provided pieces of context to answer questions. 
If you don't know the answer, just say that you don't know. 
Answer in a clinical dermatology setting with citations including text, author, section, and quote.
Question: {question} 
Context: {context} 
Answer:
"""

def main():
    download_document(URL, PATH_TO_SAVE)
    chunks = load_and_chunk_document(PATH_TO_SAVE)
    retriever = initialize_vectorstore(chunks)
    prompt = setup_prompt(TEMPLATE_STR)
    llms = initialize_llms()
    responses = query_llms(retriever, prompt, "Your question here", llms)
    for response in responses:
        print(response)

if __name__ == "__main__":
    main()
