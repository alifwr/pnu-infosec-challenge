from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.tools import tool
from dotenv import load_dotenv
import os

import sys
import time
import threading
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.loader import load_cve_cwe_data, load_personal_data


def spinner_animation(event, message="Processing..."):
    spinner = itertools.cycle(["-", "/", "|", "\\"])
    while not event.is_set():
        sys.stdout.write(f"\r{message} {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
    sys.stdout.flush()


load_dotenv()

cve_cwe_data = load_cve_cwe_data()
personal_data = load_personal_data()

docs = []
for item in cve_cwe_data:
    content = f"""
    CVE ID: {item["CVE-ID"]}
    CWE ID: {item["CWE-ID"]}
    CVSS V2: {item["CVSS-V2"]}
    CVSS V3: {item["CVSS-V3"]}
    CVSS V4: {item["CVSS-V4"]}
    Severity: {item["SEVERITY"]}
    Description: {item["DESCRIPTION"]}
    """
    document = Document(page_content=content, metadata=item)
    docs.append(document)

for item in personal_data:
    content = f"""
    Professional Persona: {item["professional_persona"]}
    Sports Persona: {item["sports_persona"]}
    Arts Persona: {item["arts_persona"]}
    Travel Persona: {item["travel_persona"]}
    Culinary Persona: {item["culinary_persona"]}
    Persona: {item["persona"]}
    Cultural Background: {item["cultural_background"]}
    Skills and Expertise: {item["skills_and_expertise"]}
    Skills and Expertise List: {item["skills_and_expertise_list"]}
    Hobbies and Interests: {item["hobbies_and_interests"]}
    Hobbies and Interests List: {item["hobbies_and_interests_list"]}
    Career Goals and Ambitions: {item["career_goals_and_ambitions"]}
    """
    document = Document(page_content=content, metadata=item)
    docs.append(document)


embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
)

vector_store = Chroma(
    collection_name="collections",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

print("Indexing Documents...")
done_event = threading.Event()
spinner_thread = threading.Thread(
    target=spinner_animation, args=(done_event, "Indexing data")
)
spinner_thread.start()

try:
    vector_store.add_documents(docs)
finally:
    done_event.set()
    spinner_thread.join()

print("Indexing Complete!")


def retrieve_documents(query: str):
    retrieved_docs = vector_store.similarity_search(query)
    return [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in retrieved_docs
    ]


@tool
def retrieve_context_tool(query: str):
    """
    Retrieve relevant context from the vector store based on the query.

    Args:
        query: The search query string.

    Returns:
        A tuple containing the serialized string of retrieved documents and the list of document objects.
    """
    retrieved_docs = vector_store.similarity_search(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized
