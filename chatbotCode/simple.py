import subprocess
from collections import deque

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

gpt_config_list = [
    {
        "model": "gpt-3.5-turbo", #can use gpt-4 too
        "api_key": "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",
    }
]

llm_config = {"config_list": gpt_config_list,"temperature": 0}


assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant answering questions about Acral Lentiginous Melanoma in a clinical setting.",
    llm_config=llm_config,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

recur_spliter = RecursiveCharacterTextSplitter(separators=["\n", "\r", "\t"])

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/wjmellon/aidiagnostics/main/acrallent.txt",
        "custom_text_split_function": recur_spliter.split_text,
    },
)


# Before sending the context to the model
problem = input("Ask a question about Acral Lentiginous Melanoma: ")
assistant_messages.append(problem)  # Add the new question to the context

# Reset the assistant before each new interaction
assistant.reset()

# If you need to process the messages (e.g., tokenize or concatenate)
context = " ".join(list(assistant_messages))  # concatenate all messages in the deque

# Pass the context to the model
ragproxyagent.initiate_chat(assistant, problem=context)

