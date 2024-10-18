from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableConfig

# Setup prompt template
def setup_prompt(template_str):
    return ChatPromptTemplate.from_template(template_str)

# Initialize two LLMs (for GPT-3.5 and GPT-4) and Llama
def initialize_llms():
    llm_1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)
    llm_2 = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
    llm_3 = ChatOllama(model="llama3.2:1b",temperature = 0, streaming=True)

    return llm_1, llm_2, llm_3
    
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
    # Initialize the LLMs
    llm_1, llm_2, llm3 = initialize_llms()
    # Build RAG chains
    rag_chain_1 = build_rag_chain(retriever, prompt, llm_1)
    rag_chain_2 = build_rag_chain(retriever, prompt, llm_2)
    rag_chain_3 = build_rag_chain(retriever, prompt, llm3)
    
    response_1 = rag_chain_1.invoke(question)
    response_2 = rag_chain_2.invoke(question)
    response_3 = rag_chain_3.invoke(question)

    return response_1, response_2, response_3

def build_llm_chain(retriever, prompt, question, llm_1, llm_2, llm3):
    # Build RAG chains
    rag_chain_1 = build_rag_chain(retriever, prompt, llm_1)
    rag_chain_2 = build_rag_chain(retriever, prompt, llm_2)
    rag_chain_3 = build_rag_chain(retriever, prompt, llm3)
    
    response_1 = rag_chain_1.invoke(question)
    response_2 = rag_chain_2.invoke(question)
    response_3 = rag_chain_3.invoke(question)

    return response_1, response_2, response_3
