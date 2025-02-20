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
        results = ddgs.text(query, max_results=10)
        return [res["href"] for res in results]

# Extract key skills and frameworks
def extract_related_skills(skill):
    web_pages = fetch_related_content(f"important aspects and related concepts of {skill}")

    extracted_data = {
        "skill_name": skill.capitalize(),
        "skill_description": "",
        "aliases": [],
        "skill_type": "",
        "skill_usages": [],
        "subsets": {
    
        }
    }

    for page in web_pages:
        print(f"Fetching data from: {page}")

        prompt = f"""
        Extract structured information about '{skill}'.
        Provide the following details strictly follow the convention provided here don't add Example: Here is the extracted information about 'java': or stars or other symbols:

        Skill Description: <brief description>
        Aliases: <comma-separated alternative names>
        Skill Type: <category like Programming Language, Farming Technique, etc.>
        Skill Usages: <comma-separated key applications>
        Subsets:
        For the dynamic subsets, first output the category name followed by a colon, 
        then on the following lines list the related items, each prefixed with "- ".
        
        For example:
        Frameworks:
        - Spring
        - Hibernate
        Libraries:
        - Apache Commons
        - Guava
        Build Tools:
        - Maven
        - Gradle
        """
        in_subsets_section = False
        response = llm.invoke(prompt)
        content = response.content.strip()
        print(content)
        current_section = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Skill Description:"):
                extracted_data["skill_description"] = line.split(": ", 1)[1]
            elif line.startswith("Aliases:"):
                extracted_data["aliases"] = [alias.strip() for alias in line.split(": ", 1)[1].split(",")]
            elif line.startswith("Skill Type:"):
                extracted_data["skill_type"] = line.split(": ", 1)[1]
            elif line.startswith("Skill Usages:"):
                extracted_data["skill_usages"] = [usage.strip() for usage in line.split(": ", 1)[1].split(",")]
            elif line.startswith("Subsets:"):
                in_subsets_section = True
                continue
            
            # Process dynamic subsets only after encountering "Subsets:"
            elif in_subsets_section:
                # If a line ends with a colon and is not a list item, treat it as a new category header
                if line.endswith(":") and not line.startswith("-"):
                    current_section = line[:-1].strip().lower().replace(" ", "_")
                    extracted_data["subsets"][current_section] = []
                # If it's a list item under the current subset, add it
                elif current_section and line.startswith("- "):
                    extracted_data["subsets"][current_section].append(line[2:])

        return extracted_data


# Extract domain-specific skills
def extract_domain_skills(domain):
    web_pages = fetch_related_content(f"essential skills for {domain}")
    skills = set()
    
    for page in web_pages:
        print(f"Fetching data from: {page}")
        
        prompt = f"""
        Extract essential skills for a career in '{domain}' from this webpage: {page}.
        
        Provide a concise bullet list of key skills with no redundant or extraneous information don't be case sensitive.
        Only include the most important keywords that are directly relevant.
        
        Format:
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
    Analyze the following query and determine its intent strictly. If the query is about a specific skill (whether IT or non-IT), classify it as 'specific skill'. If the query is about the essential skills required for a profession, job role, or industry, classify it as 'domain skills'. 

    If the query does not fit either category, respond with 'unknown'. Do not provide any explanations, just return one of these exact values:
    - specific skill
    - domain skills
    - unknown

    Query: "{query}"
    """
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    category = re.sub(r"[^a-z\s]", "", category)
    print(category)
    if category == 'specific skill':
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
