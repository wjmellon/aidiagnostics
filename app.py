import sys

from flask import Flask, request, jsonify, render_template
from data_processing.data_fetcher import download_document
from document_management.document_chunker import load_and_chunk_document
from document_management.vector_storage import initialize_vectorstore, initialize_cloud_retriever
from rag_components.generator_setup import setup_prompt, build_rag_chain
import ssl
import os
from langchain.retrievers.multi_vector import MultiVectorRetriever

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


#
# # Initialize Weaviate client
# client = weaviate.Client(
#     url="https://aichatpublic-zrouzts8.weaviate.network",  # Ensure this is your correct cloud instance URL
#     auth_client_secret=weaviate.auth.AuthApiKey(api_key="uIV5lCa7lbpj4sjDyBIrbicGquVDbWuTcZHE"),
#     additional_headers={ "X-OpenAI-Api-Key":os.environ.get('OPENAI_API_KEY', 'API key not set') }
# )


# Download and prepare data
# download_document(URL, PATH_TO_SAVE)
#chunks = load_and_chunk_document(PATH_TO_SAVE)
#chunks = split_text(PATH_TO_SAVE,100)
# retriever = initialize_cloud_retriever()
# prompt = setup_prompt(TEMPLATE_STR)
# rag_chain = build_rag_chain(retriever, prompt)



#@app.route('/ask', methods=['GET', 'POST'])
# def ask_question():
#     if request.method == 'POST':
#         # Assuming the input field in your HTML form has the name 'question'
#         question = request.form['question']
#
#         if question:
#             # Use your rag_chain to get the response
#             response = rag_chain.invoke(question)
#             # Render the same template with the question and response
#             return render_template('ask.html', question=question, answer=response)
#     # For GET requests or if question is empty, render template without response
#     return render_template('ask.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    port = request.headers.get('Database-Port', '8081')
    print(f"Received request with Database-Port: {port}")  # Debug print
    retriever = initialize_cloud_retriever(port)
    prompt = setup_prompt(TEMPLATE_STR)
    rag_chain = build_rag_chain(retriever, prompt)
    question = request.form['question']
    if question:
        # Use your rag_chain to get the response
        response = rag_chain.invoke(question)
        print(port)
        print(len(response))
        #Return question and answer in JSON format
        return jsonify(question=question, answer=response)
    return jsonify(error="No question provided"), 400

#
# @app.route('/ask', methods=['POST'])
# def ask_question():
#     port = request.headers.get('Database-Port', '8081')
#
#     print(f"Received request with Database-Port: {port}")  # Debug print
#
#     # Initialize retriever each time to ensure it uses the correct port
#     vectorretreiver = initialize_cloud_retriever(port)
#     print(f"Retriever initialized with port: {port}")  # Additional debug print
#
#     client = weaviate.Client(
#         url=f"http://localhost:{port}",
#         additional_headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')}
#     )
#     # Initialize the docstore using the custom WeaviateDocStore class
#     docstore = WeaviateDocStore(client)
#
#
#     retriever = MultiVectorRetriever(
#         vectorstore=vectorretreiver,
#         docstore=docstore,
#         id_key="doc_id"
#     )
#
#     prompt = setup_prompt(TEMPLATE_STR)
#     rag_chain = build_rag_chain(retriever, prompt)
#     question = request.form['question']
#
#     if question:
#         # Use your rag_chain to get the response
#         response = rag_chain.invoke(question)
#         print(f"Response length: {len(response)}")  # Debug print for response length
#         return jsonify(question=question, answer=response)
#
#     return jsonify(error="No question provided"), 400

# import logging
#
# #this could be useful. it transforms questions, at the moment questions are bit too radically changed.
# @app.route('/ask', methods=['POST'])
# def ask_question():
#     port = request.headers.get('Database-Port', '8081')
#     retriever = initialize_cloud_retriever(port)
#     question = request.form['question']
#
#     logging.basicConfig()
#     logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
#
#
#     if question:
#         # Use the MultiQueryRetriever to get the response
#         response = retriever.invoke(question)
#         print(response)
#         # Convert response to JSON serializable format
#         def serialize_response(response):
#             if isinstance(response, list):
#                 return [serialize_response(item) for item in response]
#             elif isinstance(response, dict):
#                 return {key: serialize_response(value) for key, value in response.items()}
#             elif hasattr(response, '__dict__'):
#                 return {key: serialize_response(value) for key, value in response.__dict__.items()}
#             elif hasattr(response, 'text'):  # Assuming 'text' attribute contains the main content
#                 return response.text
#             else:
#                 return str(response)
#
#         serialized_response = serialize_response(response)
#
#         # Return question and answer in JSON format
#         return jsonify(question=question, answer=serialized_response)
#
#     return jsonify(error="No question provided"), 400
#

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     port = request.headers.get('Database-Port', '8081')
#     retriever = initialize_cloud_retriever(port)
#     question = request.form['question']
#     llm = ChatOpenAI(temperature=0)
#     retriever_from_llm = MultiQueryRetriever.from_llm(
#         retriever=retriever, llm=llm
#     )
#
#     if question:
#         # Use your retriever to get the response
#         response = retriever_from_llm.invoke(question)
#         print(response)
#
#         # Convert response to JSON serializable format
#         def serialize_response(response):
#             if isinstance(response, list):
#                 return [serialize_response(item) for item in response]
#             elif isinstance(response, dict):
#                 return {key: serialize_response(value) for key, value in response.items()}
#             elif hasattr(response, '__dict__'):
#                 return {key: serialize_response(value) for key, value in response.__dict__.items()}
#             else:
#                 return response
#
#         serialized_response = serialize_response(response)
#
#         # Return question and answer in JSON format
#         return jsonify(question=question, answer=serialized_response)
#
#     return jsonify(error="No question provided"), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
