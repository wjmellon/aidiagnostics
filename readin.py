import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_openai.embeddings import OpenAIEmbeddings
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
    url="https://aidiag-yrjn4mqf.weaviate.network",  # Ensure this is your correct cloud instance URL
    auth_client_secret=weaviate.auth.AuthApiKey(api_key="PJ2ajygrzY7UICcaUseQxZEhl5K4NReGrLoX")
    )
test_query = client.query.get('DocumentChunk', properties=['title', 'authors', 'text', "index"]).do()
print(test_query)