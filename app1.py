import os
import tavily
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from dotenv import load_dotenv
load_dotenv()  

os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("GROQ_API_KEY")
os.getenv("NOMIC_API_KEY")

# Initialize LLM and Embeddings
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
vector_store = InMemoryVectorStore(embeddings)

# Fetch relevant pages using Tavily
def fetch_related_content(skill):
    search_results = tavily.search(f"related skills and frameworks for {skill}")
    web_paths = [result["url"] for result in search_results]
    return web_paths

# Process and extract related skills
def extract_related_skills(skill):
    web_pages = fetch_related_content(skill)
    related_skills = {}
    
    for page in web_pages:
        response = llm.invoke(f"Extract related skills and frameworks for {skill} from this page: {page}")
        related_skills[page] = response.content
    
    return related_skills

# Execute
skill_query = "Java"
related_skills = extract_related_skills(skill_query)
print(related_skills)
