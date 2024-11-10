from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOllama


def setup_prompt(template_str):
    """Set up the chat prompt template."""
    return ChatPromptTemplate.from_template(template_str)

def initialize_llms():
    """Initialize multiple LLMs for querying."""
    llm_1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_2 = ChatOpenAI(model="gpt-4", temperature=0)
    llm_3 = ChatOllama(model="llama3.2:1b", temperature=0.5)
    # gemma google model
    llm_4 = ChatOllama(model="gemma2:2b", temperature=0.5)
    
    return llm_1, llm_2, llm_3, llm_4

def build_rag_chain(retriever, prompt, llm):
    """Build a Retrieval-Augmented Generation (RAG) chain."""
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def query_llms(retriever, prompt, question, llms=None):
    """Query multiple LLMs synchronously."""
    if llms is None:
        llms = initialize_llms()
    responses = []
    for llm in llms:
        rag_chain = build_rag_chain(retriever, prompt, llm)
        response = rag_chain.invoke(question)
        responses.append(response)
    return responses
