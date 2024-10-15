import sys
from flask import Flask, request, render_template
from document_management.vector_storage import initialize_port_retriever
from rag_components.generator_setup import setup_prompt, initialize_llms, build_llm_chain
import ssl
import os

# Create Flask app
app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('ask.html')

class LLM:
    def __init__(self, model1, model2, model3):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def get_llm1(self):
        return self.model1
    
    def get_llm2(self):
        return self.model2
    
    def get_llm3(self):
        return self.model3

# Disable SSL warnings (Note: only for development, not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Print out the OpenAI API key for verification (Note: Be cautious about printing sensitive information in production environments)
print(os.environ.get('OPENAI_API_KEY', 'API key not set'))

# Constants and configurations
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
# retriever = initialize_port_retriever()
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

def ask_question(llm_instance, retriever, prompt):
    question = input("Please enter your question: ")

    if question:
        # Call the function to query both LLMs synchronously
        response_1, response_2, response_3 = build_llm_chain(
            retriever, 
            prompt, 
            question, 
            llm_instance.get_llm1(), 
            llm_instance.get_llm2(), 
            llm_instance.get_llm3()
        )

        # Combine the responses (you can modify how you combine them)
        combined_response = f"LLM 1 response: {response_1}\nLLM 2 response: {response_2}\nLLM 3 response: {response_3}"

        print(f"Response Length: {len(combined_response)}")
        print(f"Combined Response:\n{combined_response}")
    else:
        print("No question provided.")

if __name__ == '__main__':
    port = input("Enter Database-Port (default: 8081): ") or '8081'

    # Initialize the retriever
    retriever = initialize_port_retriever(port)

    # Setup the prompt
    prompt = setup_prompt(TEMPLATE_STR)

    # Initialize LLMs and create LLM instance
    llm1, llm2, llm3 = initialize_llms()
    llm_instance = LLM(llm1, llm2, llm3)

    print(f"Using Database Port: {port}")

    while True:
        ask_question(llm_instance, retriever, prompt)
        cont = input("Do you want to ask another question? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Exiting...")
            break