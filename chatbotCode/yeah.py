import os
import requests
import ssl
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate  # Adjust the path if it's different
import warnings
warnings.filterwarnings('ignore')



# Optional: Disable SSL certificate validation (Note: This is not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Print OpenAI API key to ensure it's set correctly (Remove this in production for security reasons)
print(os.environ['OPENAI_API_KEY'])

# Load the document from the URL
url = "https://raw.githubusercontent.com/wjmellon/aidiagnostics/main/aggregated.txt"
res = requests.get(url)
with open("aggregated.txt", "w") as f:
    f.write(res.text)

# Load the document using TextLoader
loader = TextLoader('./aggregated.txt')
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Initialize Weaviate client and populate vectorstore
client = weaviate.Client(embedded_options=EmbeddedOptions())
vectorstore = Weaviate.from_documents(
    client=client,
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    by_text=False
)

# Define the retriever component
retriever = vectorstore.as_retriever()

# Prepare a prompt template for augmenting the prompt with additional context
template = """You are an assistant for question-answering tasks. These questions are about Acral Lentiginous Melanoma.
You must use the provided pieces of context to answer questions. 
If you don't know the answer, just say that you don't know. 
Answer in a clinical dermatology setting. Give the user a citation from the text, author, section, and quote from text.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the LLM (Language Learning Model)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Build the RAG (Retrieval-Augmented Generation) chain
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
query = ""
while query != "exit":
    # Invoke the RAG chain with a query
    query = input("Ask a question about Acral Lentiginous Melanoma or type 'exit' to quit: ")
    response = rag_chain.invoke(query)

    # Print the response
    print(response)
