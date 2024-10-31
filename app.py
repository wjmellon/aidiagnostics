import os
import ssl
import json
from document_management.vector_storage import initialize_port_retriever
from rag_components.generator_setup import setup_prompt, initialize_llms, query_llms
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def log_responses(question, responses, file_path='response.json'):
    data = {
        "question": question,
        "responses": {
            "LLM_1": responses[0],
            "LLM_2": responses[1],
            "LLM_3": responses[2]
        }
    }
    try:
        with open(file_path, 'r+') as f:
            file_data = json.load(f)
            file_data.append(data)
            f.seek(0)
            json.dump(file_data, f, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(file_path, 'w') as f:
            json.dump([data], f, indent=4)

def main():
    # Ensure SSL context is set up properly
    ssl._create_default_https_context = ssl._create_unverified_context

    # Prompt the user for a question
    question = input("Enter your question about skin cancer: ").strip()
    if question:
        # Retrieve the port from environment variables or default to '8081'
        port = os.getenv('WEAVIATE_PORT', '8081')
        retriever = initialize_port_retriever(port)

        # Define the prompt template
        prompt_template = """You are an assistant for question-answering tasks. These questions are about skin cancer.
You must use the provided pieces of context to answer questions. If you don't know the answer, just say that you don't know.
Provide detailed clinical answers, citing sources with quotes, authors, title, and URL.
Always put the DOI URL at the end in an HTML tag <a href="putlinkhere" target="_blank">Link</a>.
Question: {question} Context: {context} Answer:"""

        # Set up the prompt and initialize language models
        prompt = setup_prompt(prompt_template)
        llms = initialize_llms()

        # Query the language models
        try:
            responses = query_llms(retriever, prompt, question, llms)
            log_responses(question, responses)

            # Print the responses to the terminal
            print("\nResponses:")
            for idx, response in enumerate(responses, start=1):
                print(f"LLM_{idx} Response:\n{response}\n")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No question provided.")

if __name__ == '__main__':
    main()
