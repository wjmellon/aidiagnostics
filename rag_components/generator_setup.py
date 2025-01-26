from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def setup_prompt(template_str):
    """Set up the chat prompt template."""
    return ChatPromptTemplate.from_template(template_str)

def initialize_llms():
    """Initialize multiple LLMs for querying."""
    callbacks = [StreamingStdOutCallbackHandler()]
    
    llm_1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=callbacks)
    llm_2 = ChatOpenAI(model="gpt-4", temperature=0, streaming=True, callbacks=callbacks)
    llm_3 = ChatOllama(model="llama3.2:1b", temperature=0.5, callbacks=callbacks)
    llm_4 = ChatOllama(model="gemma2:2b", temperature=0.5, callbacks=callbacks)
    
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
    
    for i, llm in enumerate(llms):
        model_name = ""
        if i == 0:
            print("🤖 GPT-3.5 RESPONSE")
        elif i == 1:
            print("🧠 GPT-4 RESPONSE")
        elif i == 2:
            print("🦙 LLAMA RESPONSE")
        else:
            print("💎 GEMMA RESPONSE")
            
        rag_chain = build_rag_chain(retriever, prompt, llm)
        response = rag_chain.invoke(question)
        responses.append(response)
        print("\n" + "="*80 + "\n")
    return responses
