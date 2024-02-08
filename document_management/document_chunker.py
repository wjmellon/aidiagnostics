
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


def load_and_chunk_document(path, chunk_size=500, chunk_overlap=50):
    loader = TextLoader("data/aggregated.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
