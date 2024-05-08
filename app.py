import sys

from flask import Flask, request, jsonify, render_template
from data_processing.data_fetcher import download_document
from document_management.document_chunker import load_and_chunk_document
from document_management.vector_storage import initialize_vectorstore
from rag_components.generator_setup import setup_prompt, build_rag_chain
import ssl
import os
import spacy


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
URL = "https://github.com/wjmellon/aidiagnostics/blob/main/data/aggregated.txt"
PATH_TO_SAVE = "./data/collected_texts.txt"
TEMPLATE_STR = """You are an assistant for question-answering tasks. These questions are about skin cancer. You must use the provided pieces of context to answer questions. If you don't know the answer, just say that you don't know. Answer in a clinical dermatology setting. You have to give the user a citation from the text, author, section, and quote from text. You MUST give the quote and the authors from context. Question: {question} Context: {context}Answer:"""

# Download and prepare data
# download_document(URL, PATH_TO_SAVE)
chunks = load_and_chunk_document(PATH_TO_SAVE)
#chunks = split_text(PATH_TO_SAVE,100)
retriever = initialize_vectorstore(chunks)
prompt = setup_prompt(TEMPLATE_STR)
rag_chain = build_rag_chain(retriever, prompt)



@app.route('/ask', methods=['GET', 'POST'])
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
    question = request.form['question']
    if question:
        # Use your rag_chain to get the response
        response = rag_chain.invoke(question)
        # Return question and answer in JSON format
        return jsonify(question=question, answer=response)
    return jsonify(error="No question provided"), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
