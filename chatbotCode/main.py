import subprocess
from collections import deque
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
print(os.environ['OPENAI_API_KEY'])
API_KEY = os.environ['OPENAI_API_KEY']


import requests
from langchain.document_loaders import TextLoader

url = "https://raw.githubusercontent.com/wjmellon/aidiagnostics/main/aggregated.txt"
res = requests.get(url)
with open("aggregated.txt", "w") as f:
    f.write(res.text)

loader = TextLoader('./aggregated.txt')
documents = loader.load()


from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
  embedded_options = EmbeddedOptions()
)
vectorstore = Weaviate.from_documents(
    client = client,
    documents = chunks,
    embedding = OpenAIEmbeddings(),
    by_text = False
)
retriever = vectorstore.as_retriever()

# Define the maximum number of exchanges (question + answer pairs) to keep
MAX_EXCHANGES = 4

# Initialize a deque to store the exchanges with a maximum length
assistant_messages = deque(maxlen=MAX_EXCHANGES * 2)

import autogen
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the truncate_context function
# Debugging function to print the length of the context
def debug_print_context_length(context, tokenizer):
    tokenized_length = len(tokenizer.encode(context))
    print(f"Current context length in tokens: {tokenized_length}")

# Define the truncate_context function
def truncate_context(messages, tokenizer, max_tokens):
    total_length = len(tokenizer.encode(" ".join(messages)))
    debug_print_context_length(" ".join(messages), tokenizer)  # Debug print
    while total_length > max_tokens and messages:
        messages.popleft()
        total_length = len(tokenizer.encode(" ".join(messages)))
        debug_print_context_length(" ".join(messages), tokenizer)  # Debug print

gpt_config_list = [
    {
        "model": "gpt-3.5-turbo", #can use gpt-4 too. THIS COST MORE MONEY ONLY DO THIS FOR DEMO PLEASE ASK WALKER FIRST DO NO BANKRUPT HIM
        "api_key": API_KEY,
    }
]

llm_config = {"config_list": gpt_config_list,"temperature": 0}


assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant answering questions about Acral Lentiginous Melanoma in a clinical setting. When you answer questions refer to the context given and provide citation from context.",
    llm_config=llm_config,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

recur_spliter = RecursiveCharacterTextSplitter(separators=["\n", "\r", "\t"])

from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=API_KEY,
                model_name="text-embedding-ada-002"
            )

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": retriever,
        "custom_text_split_function": recur_spliter.split_text,
        "embedding_function": openai_ef,
        "chunk_token_size": 1000
    },
)
problem = input("Ask a question about Acral Lentiginous Melanoma: ")
assistant.reset()
ragproxyagent.initiate_chat(assistant, problem=problem)


# Check token length before adding the new message
# potential_length = len(tokenizer.encode(" ".join(list(assistant_messages)) + " " + problem))
# if potential_length <= MAX_TOKENS:
#     assistant_messages.append(problem)
# else:
#     truncate_context(assistant_messages, tokenizer, MAX_TOKENS - len(tokenizer.encode(problem)))
#     assistant_messages.append(problem)
#
# # Forcefully ensure the context length is within limits before sending it to the model
# truncate_context(assistant_messages, tokenizer, MAX_TOKENS)
# context = " ".join(list(assistant_messages))  # concatenate all messages in the deque
# debug_print_context_length(context, tokenizer)  # Debug print
#
# # If the context is still too long, truncate it to a fixed size (last resort)
# if len(tokenizer.encode(context)) > MAX_TOKENS:
#     print("Context is still too long, forcefully shortening")
#     context = tokenizer.decode(tokenizer.encode(context)[:MAX_TOKENS])
#
# # Pass the context to the model
# assistant.reset()
# ragproxyagent.initiate_chat(assistant, problem=context)