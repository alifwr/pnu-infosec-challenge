from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.tools import tool
from dotenv import load_dotenv
import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_cve_cwe_data, load_personal_data

load_dotenv()

cve_cwe_data = load_cve_cwe_data()
personal_data = load_personal_data()

docs = []
for item in cve_cwe_data:
    content = item["DESCRIPTION"]
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


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

vector_store = Chroma(
    collection_name="collections",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

vector_store.add_documents(docs)
vector_store.persist()

# Example queries
print(vector_store.similarity_search("What is CVE-2024-4655?"))


# @tool
# def retrieve(query: str):
#     retriever = vector_store.as_retriever()
#     return retriever.invoke(query)
