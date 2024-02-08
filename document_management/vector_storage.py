

import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_openai.embeddings import OpenAIEmbeddings
from weaviate.embedded import EmbeddedOptions

def initialize_vectorstore(chunks):
    client = weaviate.Client(embedded_options=EmbeddedOptions())
    vectorstore = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        by_text=False
    )
    return vectorstore.as_retriever()
