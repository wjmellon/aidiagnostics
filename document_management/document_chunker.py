from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# look into best chunking strategy

def load_and_chunk_document(path, chunk_size=300, chunk_overlap=50):
    with open(path, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()  # Read the entire content of the file

        # Now write the content back to the same file in UTF-8 encoding
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)