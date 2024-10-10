import sys
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import asyncio

# Setup prompt template
def setup_prompt(template_str):
    return ChatPromptTemplate.from_template(template_str)

# Initialize two LLMs (for GPT-3.5 and GPT-4)
def initialize_llms():
    llm_1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm_2 = ChatOpenAI(model_name="gpt-4", temperature=0)
    return llm_1, llm_2

# Build the RAG chain for each LLM
def build_rag_chain(retriever, prompt, llm):
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to query two LLMs concurrently
# Function to query two LLMs synchronously
def query_llms_concurrently(retriever, prompt, question):
    llm_1, llm_2 = initialize_llms()

    # Build RAG chains for both LLMs
    rag_chain_1 = build_rag_chain(retriever, prompt, llm_1)
    rag_chain_2 = build_rag_chain(retriever, prompt, llm_2)

    # Invoke both LLMs synchronously (no asyncio needed)
    response_1 = rag_chain_1.invoke(question)
    response_2 = rag_chain_2.invoke(question)

    return response_1, response_2

