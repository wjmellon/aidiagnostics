import sys

#temperature = 0 means more precision because its a lower number

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

#temperature = 0 means more precision because its a lower number

def setup_prompt(template_str):
    return ChatPromptTemplate.from_template(template_str)

def initialize_llm(model_name="gpt-3.5-turbo", temperature=0):
    return ChatOpenAI(model_name=model_name, temperature=temperature)

def build_rag_chain(retriever, prompt):
    llm = initialize_llm()
    return (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser(metadata_keys=["paper_title", "authors"])
    )