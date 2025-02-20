import os
import json
import re
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
    web_pages = fetch_related_content(f"related skills, frameworks, libraries, and tools for {skill}")
    
    extracted_data = {
        "skill_name": skill.capitalize(),
        "skill_description": "",
        "aliases": [],
        "skill_type": "",
        "skill_usages": [],
        "subsets": {
            "frameworks": [],
            "libraries": [],
            "build_tools": []
        }
    }

    for page in web_pages:
        print(f"Fetching data from: {page}")

        prompt = f"""
        Extract detailed information about '{skill}' from this webpage: {page}

        Format of the response is this and dont add stars before any text:
        Skill Name: Java
        Skill Description: Java is a widely-used programming language...
        Aliases: java, java 8
        Skill Type: Programming Language
        Skill Usages: Software Development, Android App Development

        Frameworks: Spring, Hibernate
        Libraries: Apache Commons, Guava
        Build Tools: Maven, Gradle
        """

        response = llm.invoke(prompt)
        content = response.content.strip()
        print("\nRaw LLM Response:\n", content)  # Debugging output

        # Parsing the response
        current_section = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Skill Name:"):
                extracted_data["skill_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Skill Description:"):
                extracted_data["skill_description"] = line.split(":", 1)[1].strip()
            elif line.startswith("Aliases:"):
                extracted_data["aliases"] = [alias.strip() for alias in line.split(":", 1)[1].split(",")]
            elif line.startswith("Skill Type:"):
                extracted_data["skill_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("Skill Usages:"):
                extracted_data["skill_usages"] = [usage.strip() for usage in line.split(":", 1)[1].split(",")]
            elif line.startswith("Frameworks:"):
                extracted_data["subsets"]["frameworks"] = [fw.strip() for fw in line.split(":", 1)[1].split(",")]
            elif line.startswith("Libraries:"):
                extracted_data["subsets"]["libraries"] = [lib.strip() for lib in line.split(":", 1)[1].split(",")]
            elif line.startswith("Build Tools:"):
                extracted_data["subsets"]["build_tools"] = [bt.strip() for bt in line.split(":", 1)[1].split(",")]
    return extracted_data


# Extract domain-specific skills
def extract_domain_skills(domain):
    web_pages = fetch_related_content(f"essential skills for {domain}")
    skills = set()
    
    for page in web_pages:
        print(f"Fetching data from: {page}")
        
        prompt = f"""
        Extract essential skills for a career in '{domain}' from this webpage: {page}
        
        Format the response like this whatever maybe the format of query:
        
        Required Skills:
        - Skill1
        - Skill2
        - Skill3
        """
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        for line in content.split("\n"):
            if line.startswith("- "):
                skills.add(line[2:])
    
    return {"Domain": domain, "Required Skills": list(skills)}

# Supervisor Agent
def supervisor_agent(query):
    prompt = f"""
    Analyze the following query and determine its intent strictly. If the query is about technology-related skills, programming languages, or frameworks, classify it as 'related skills'. If it is about the necessary skills for a profession, job role, or industry (even if non-IT), classify it as 'domain skills'. 

    If the query does not fit either category, respond with 'unknown'. Do not provide any explanations, just return one of these exact values:
    - related skills
    - domain skills
    - unknown

    Query: "{query}"
    """
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    category = re.sub(r"[^a-z\s]", "", category)
    print(category)
    if category == 'related skills':
        skill = query.split("for")[-1].strip()
        return extract_related_skills(skill)
    elif category == 'domain skills':
        domain = query.split("for")[-1].strip()
        return extract_domain_skills(domain)
    else:
        return {"Error": "Unable to classify the query."}

# Execute
query = input("Enter your query: ")
result = supervisor_agent(query)

# Print JSON output
print(json.dumps(result, indent=4))
