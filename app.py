import sys

from flask import Flask, request, jsonify, render_template
from data_processing.data_fetcher import download_document
from document_management.document_chunker import load_and_chunk_document
from document_management.vector_storage import initialize_vectorstore, initialize_port_retriever
from rag_components.generator_setup import setup_prompt, build_rag_chain,query_llms_concurrently
import ssl
import os
from langchain.retrievers.multi_vector import MultiVectorRetriever
import asyncio


import weaviate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI



# Create Flask app
app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('ask.html')

# Disable SSL warnings (Note: only for development, not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Print out the OpenAI API key for verification (Note: Be cautious about printing sensitive information in production environments)
print(os.environ.get('OPENAI_API_KEY', 'API key not set'))

# Constants and configurations
#URL = "https://github.com/wjmellon/aidiagnostics/blob/main/data/aggregated.txt"
#PATH_TO_SAVE = "./data/collected_texts.txt"
TEMPLATE_STR = """You are an assistant for question-answering tasks. These questions are about skin cancer. You must use the provided pieces of context to answer questions. Do your best to contextualize the answer with the question. You're using context to answer these questions. The user is not asking the documents, but you are referring to the context to answer these questions. If you don't know the answer, just say that you don't know. Answer with medium detail in a clinical setting. You have to give the user a citation from the text, author, section, and quote from text. You MUST give the quote,authors, title, and url from context. Give the source of your answer in a quotation. You need to be exact. Dont give references to general question with answers about specific concepts, references HAVE to match answers well. You have some freedom in deciding what context is relevant to the question, sometimes documents will be not relevant to user query. Always put doi_url at end in quotes in html tag <a href = "putlinkhere target="_blank">Link</a> for javascript. Question: {question} Context: {context}Answer:"""


 #This is the original
# @app.route('/ask', methods=['POST'])
# def ask_question():
#     port = request.headers.get('Database-Port', '8081')
#     print(f"Received request with Database-Port: {port}")  # Debug print
#     retriever = initialize_cloud_retriever(port)
#     prompt = setup_prompt(TEMPLATE_STR)
#     rag_chain = build_rag_chain(retriever, prompt)
#     question = request.form['question']
#     if question:
#         # Use your rag_chain to get the response
#         response = rag_chain.invoke(question)
#         print(port)
#         print(len(response))
#         #Return question and answer in JSON format
#         return jsonify(question=question, answer=response)
#     return jsonify(error="No question provided"), 400

def ask_question():
    # Get the Database-Port (simulate it, e.g., default to '8081')
    port = input("Enter Database-Port (default: 8081): ") or '8081'
    print(f"Received request with Database-Port: {port}")  # Debug print

    # Initialize the retriever (assuming it's defined elsewhere)
    retriever = initialize_port_retriever(port)  # Adjust based on your retriever setup

    # Setup the prompt (use TEMPLATE_STR from your context)
    prompt = setup_prompt(TEMPLATE_STR)

    # Get the question from the terminal input
    question = input("Please enter your question: ")

    if question:
        # Call the function to query both LLMs synchronously
        response_1, response_2, response3 = query_llms_concurrently(retriever, prompt, question)

        # Combine the responses (you can modify how you combine them)
        combined_response = f"LLM 1 response: {response_1}\nLLM 2 response: {response_2}\nLlama response: {response3}"

        print(f"Database Port: {port}")
        print(f"Response Length: {len(combined_response)}")
        print(f"Combined Response:\n{combined_response}")
    else:
        print("No question provided.")


if __name__ == '__main__':
    # Loop to keep asking questions until the user exits
    while True:
        ask_question()
        cont = input("Do you want to ask another question? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Exiting...")
            break
