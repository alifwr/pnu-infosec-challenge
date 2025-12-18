from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3:8b")


def init_chat_model():
    return model
