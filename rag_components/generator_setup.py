# import sys
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# import asyncio
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import transformers
# import torch

# # Setup prompt template
# def setup_prompt(template_str):
#     return ChatPromptTemplate.from_template(template_str)

# # Initialize two LLMs (for GPT-3.5 and GPT-4) and Llama
# def initialize_llms():
#     # initialize GPT-3.5 and GPT-4
#     llm_1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#     llm_2 = ChatOpenAI(model_name="gpt-4", temperature=0)


#     # Initialize Llama
#     model_id = "meta-llama/Llama-3.2-1B"
#     llama_model = pipeline(
#         # model type
#         "text-generation", 
#         model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16), 
#         tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B"),
#         device_map="auto"
#     )

#     return llm_1, llm_2, llama_model

# # RAG chain builder
# def build_rag_chain(retriever, prompt, llm):
#     return (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )


# # Function to query 3 LLMs synchronously
# def query_llms_concurrently(retriever, prompt, question):
#     llm_1, llm_2, llama_model = initialize_llms()

#     # Build RAG chains for both LLMs
#     rag_chain_1 = build_rag_chain(retriever, prompt, llm_1)
#     rag_chain_2 = build_rag_chain(retriever, prompt, llm_2)
    
#     llama_response = llama_model(question, max_length=256, truncation=True)

#     # Invoke both LLMs synchronously (no asyncio needed)
#     response_1 = rag_chain_1.invoke(question)
#     response_2 = rag_chain_2.invoke(question)
#     response_3 = llama_response[0]['generated_text']

#     return response_1, response_2, response_3

import sys
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch

# Setup prompt template
def setup_prompt(template_str):
    return ChatPromptTemplate.from_template(template_str)

# Initialize two LLMs (for GPT-3.5 and GPT-4) and Llama
def initialize_llms():
    print("Initializing GPT-3.5 and GPT-4...")
    llm_1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm_2 = ChatOpenAI(model_name="gpt-4", temperature=0)
    print("GPT models initialized.")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Initialize Llama
    print("Initializing LLaMA model...")
    model_id = "meta-llama/Llama-3.2-1B"
    llama_model = pipeline(
        "text-generation", 
        model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16), 
        tokenizer=tokenizer,
        device="cpu"
    )
    print("LLaMA model initialized.")

    return llm_1, llm_2, llama_model

# RAG chain builder
def build_rag_chain(retriever, prompt, llm):
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to query 3 LLMs synchronously
def query_llms_concurrently(retriever, prompt, question):
    print("Initializing models...")
    llm_1, llm_2, llama_model = initialize_llms()
    print("Models initialized successfully.")

    # Build RAG chains for both GPT models
    print("Building RAG chains for GPT models...")
    rag_chain_1 = build_rag_chain(retriever, prompt, llm_1)
    rag_chain_2 = build_rag_chain(retriever, prompt, llm_2)
    print("RAG chains built.")

    # Invoke both GPT models synchronously (no asyncio needed)
    print(f"Querying GPT-3.5 with the question: {question}")
    response_1 = rag_chain_1.invoke(question)
    print("GPT-3.5 query completed.")

    print(f"Querying GPT-4 with the question: {question}")
    response_2 = rag_chain_2.invoke(question)
    print("GPT-4 query completed.")

    # Query LLaMA model
    print("Querying LLaMA model...")
    llama_response = llama_model(question, max_length=256)
    print(f"LLaMA response: {llama_response[0]['generated_text']}")

    response_3 = llama_response[0]['generated_text']
    print("All queries completed.")

    return response_1, response_2, response_3
