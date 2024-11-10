import os
import ssl
import json
from document_management.vector_storage import initialize_port_retriever
from rag_components.generator_setup import setup_prompt, initialize_llms, query_llms
from dotenv import load_dotenv
import csv
import pandas as pd

# Load environment variables from .env file
load_dotenv()
    
def main():
    # Ensure SSL context is set up properly
    ssl._create_default_https_context = ssl._create_unverified_context

    # Prompt the user for a question
    question = input("Enter your question about skin cancer: ").strip()
    if question:
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
            print("LLM 1: ", responses[0])
            print("LLM 2: ", responses[1])
            print("LLM 3: ", responses[2])
            print("LLM 4: ", responses[3])

            # Initialize CSV file if it doesn't exist
            if not os.path.exists('response.csv'):
                with open('response.csv', 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['question', 'LLM_1', 'LLM_2', 'LLM_3', 'LLM_4'])
                
            # Strip any potential command artifacts from the response
            # for i in range(len(responses)):
            #     responses[i] = responses[i].replace(",", "").strip()
                    
            # Create a dictionary with the question and responses
            data = {
                'question': [question],
                'LLM_1': [responses[0]],
                'LLM_2': [responses[1]], 
                'LLM_3': [responses[2]],
                'LLM_4': [responses[3]]
            }
            
            # Write the data df to the CSV file
            df = pd.DataFrame(data)
            df.to_csv('response.csv', mode='a', header=False, index=False)

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No question provided.")

if __name__ == '__main__':
    main()
