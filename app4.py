import os
import json
from duckduckgo_search import DDGS
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()  

os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("GROQ_API_KEY")
os.getenv("NOMIC_API_KEY")

# Initialize LLM
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
        
        Format the response as JSON with 'Skills' and 'Frameworks' keys. Please don't add any other extra info i just want skills and framework
        """

        response = llm.invoke(prompt)
        content = response.content.strip()
        print(content)
        try:
            parsed_response = json.loads(content)
            extracted_skills["Skills"].update(parsed_response.get("Skills", []))
            extracted_skills["Frameworks"].update(parsed_response.get("Frameworks", []))
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON")
    
    return {key: list(values) for key, values in extracted_skills.items()}

# Extract domain-specific skills
def extract_domain_skills(domain):
    web_pages = fetch_related_content(f"essential skills for {domain}")
    skills = set()
    
    for page in web_pages:
        print(f"Fetching data from: {page}")
        
        prompt = f"""
        Extract essential skills for a career in '{domain}' from this webpage: {page}
        
        Format the response as JSON with 'Required Skills' key.
        """
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        try:
            parsed_response = json.loads(content)
            skills.update(parsed_response.get("Required Skills", []))
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON")
    
    return {"Domain": domain, "Required Skills": list(skills)}

# Supervisor Agent
def supervisor_agent(query):
    prompt = f"""
    Analyze the following query and determine its intent. If the query is about skills related to a specific technology, programming language, or field of expertise, classify it as 'related skills'. If the query is about the required skills for a profession, job role, or industry, classify it as 'domain skills'.
    
    Query: "{query}"
    
    Respond with only 'related skills' or 'domain skills'.
    """
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    
    if (True):
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

# Print JSON output
print(json.dumps(result, indent=4))
