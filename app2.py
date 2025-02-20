import os
import json
from duckduckgo_search import DDGS
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

# Fetch relevant pages using DuckDuckGo
def fetch_related_content(skill):
    with DDGS() as ddgs:
        results = ddgs.text(f"related skills and frameworks for {skill}", max_results=5)
        return [res["href"] for res in results]

# Process and extract related skills
def extract_related_skills(skill):
    web_pages = fetch_related_content(skill)
    extracted_skills = {"Skills": set(), "Frameworks": set()}
    
    for page in web_pages:
        print(f"Fetching data from: {page}")  # Debugging output

        prompt = f"""
        Extract key skills and frameworks related to '{skill}' from this webpage: {page}
        
        Format the response like this:
        
        Skills:
        - Skill1
        - Skill2
        - Skill3
        
        Frameworks:
        - Framework1
        - Framework2
        - Framework3
        """

        response = llm.invoke(prompt)
        content = response.content.strip()
        print("\nRaw LLM Response:\n", content)  # Debugging output

        # Extract skills and frameworks
        current_section = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Skills:"):
                current_section = "Skills"
                continue
            elif line.startswith("Frameworks:"):
                current_section = "Frameworks"
                continue

            if current_section and line.startswith("- "):  # Collect only bullet points
                extracted_skills[current_section].add(line[2:])  # Remove "- " from each line

    return {key: list(values) for key, values in extracted_skills.items()}  # Convert sets to lists

# Execute
skill_query = "Java"
related_skills = extract_related_skills(skill_query)

# Print formatted output
print("\n=== Related Skills and Frameworks ===")
print("Skills:", ",\n ".join(related_skills.get("Skills", [])))
print("Frameworks:", ",\n ".join(related_skills.get("Frameworks", [])))