import os
import bs4
import networkx as nx
import matplotlib.pyplot as plt
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# API Keys
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"
os.environ["NOMIC_API_KEY"] = "nk-XhPkgqbU9v_IK2wiRwLWT2JyZdoSK-Gw106X_GcaGlA"

# Initialize LLM and Embeddings
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
vector_store = InMemoryVectorStore(embeddings)

# Load resumes from the web
resume_sources = [
    "https://drive.google.com/file/d/1zm0WRGBobgjbwv1EHTc3rqRoj4qz1348/view"   
]
loader = WebBaseLoader(web_paths=resume_sources)
docs = loader.load()

# Process resumes
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store.add_documents(documents=all_splits)

# Define State
class State(TypedDict):
    job_description: str
    context: List[str]
    extracted_skills: dict

# Retrieve relevant resumes
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["job_description"])
    return {"context": [doc.page_content for doc in retrieved_docs]}

# Extract skills
def extract_skills(state: State):
    extracted = {}
    for doc in state["context"]:
        response = llm.invoke(f"Extract skills and sub-skills: {doc}")
        extracted[doc] = response.content
    return {"extracted_skills": extracted}

from pyvis.network import Network 

# Visualize skills as a graph
def visualize_skills(state: State):
    net = Network(notebook=True, height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    for resume, skills in state["extracted_skills"].items():
        net.add_node(resume, label=resume, color='blue')
        for skill in skills.split(","):
            skill = skill.strip()
            net.add_node(skill, label=skill, color='green')
            net.add_edge(resume, skill)

    net.show("skills_graph.html") 
    
    return {}

# Build Graph
graph_builder = StateGraph(State).add_sequence([retrieve, extract_skills, visualize_skills])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Execute
response = graph.invoke({"job_description": "Looking for a Python developer with experience in ML"})