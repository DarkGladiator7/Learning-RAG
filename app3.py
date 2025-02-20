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

#Initialize LLM
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# Fetch relevant pages using DuckDuckGo
def fetch_related_content(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return [res["href"] for res in results]

# Extract key skills and frameworks
def extract_related_skills(skill):
    web_pages = fetch_related_content(f"related skills and frameworks for {skill}")
    extracted_skills = {"Skills": set(), "Frameworks": set()}
    
    for page in web_pages:
        print(f"Fetching data from: {page}")

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
        #print("\nRaw LLM Response:\n", content)

        current_section = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Skills:"):
                current_section = "Skills"
                continue
            elif line.startswith("Frameworks:"):
                current_section = "Frameworks"
                continue

            if current_section and line.startswith("- "):
                extracted_skills[current_section].add(line[2:])

    return {key: list(values) for key, values in extracted_skills.items()}

# Extract domain-specific skills
def extract_domain_skills(domain):
    web_pages = fetch_related_content(f"essential skills for {domain}")
    skills = set()
    
    for page in web_pages:
        print(f"Fetching data from: {page}")
        
        prompt = f"""
        Extract essential skills for a career in '{domain}' from this webpage: {page}
        
        Format:
        
        Required Skills:
        - Skill1
        - Skill2
        - Skill3
        """
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        #print("\nRaw LLM Response:\n", content)

        for line in content.split("\n"):
            if line.startswith("- "):
                skills.add(line[2:])
    
    return {"Domain": domain, "Required Skills": list(skills)}

# Supervisor Agent
def supervisor_agent(query):
    prompt = f"""
    Classify the following query as either 'related skills' or 'domain skills':
    
    Query: "{query}"
    
    Respond with only 'related skills' or 'domain skills'.
    """
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    
    if category == "related skills":
        skill = query.split("for")[-1].strip()
        return extract_related_skills(skill)
    elif category == "domain skills":
        domain = query.split("for")[-1].strip()
        return extract_domain_skills(domain)
    else:
        return {"Error": "Unable to classify the query."}

# Execute
query = input("Enter your query: ")
result = supervisor_agent(query)

# Print formatted output
if "Error" in result:
    print(result["Error"])
else:
    if "Domain" in result:
        print("\n=== Required Skills for Domain ===\n")
        print("Domain:", result["Domain"])
        print("Skills:", ", ".join(result["Required Skills"]))
    else:
        print("\n=== Related Skills and Frameworks ===\n")
        skills_list = result.get("Skills", [])
        frameworks_list = result.get("Frameworks", [])

        max_len = max(len(skills_list), len(frameworks_list))
        skills_list += [""] * (max_len - len(skills_list))
        frameworks_list += [""] * (max_len - len(frameworks_list))

        table = list(zip(skills_list, frameworks_list))
        print(tabulate(table, headers=["Skills", "Frameworks"], tablefmt="grid"))