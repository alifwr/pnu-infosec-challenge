from langchain_ollama import ChatOllama

from tools.rag import retrieve_context

res = retrieve_context.invoke("hello naruto")
print("RES: ", res)
model = ChatOllama(model="qwen3:8b")

result = model.invoke("hello")

print(result)
