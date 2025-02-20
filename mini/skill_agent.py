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

'''def fetch_related_content(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        return [res["href"] for res in results]'''
import time
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

def fetch_related_content(query, max_results=5, retries=3, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                return [res["href"] for res in results]
        except DuckDuckGoSearchException as e:
            print(f"Attempt {attempt+1} failed with error: {e}")
            attempt += 1
            time.sleep(delay * attempt)  # Exponential backoff
    raise Exception("Exceeded maximum retries for fetching related content.")

def extract_related_skills(skill):
    # Use a dynamic query that works across domains
    web_pages = fetch_related_content(f"{skill} skill details")
    extracted_data = {
        "skill_name": skill.capitalize(),
        "skill_description": "",
        "aliases": [],
        "skill_type": "",
        "skill_usages": [],
        "subsets": {}
    }
    
    prompt = f"""
    Extract structured information about '{skill}'.
    Provide the following details in the same format don't add any extra symbols or stars in the beginning or at the end:
    
    Skill Description: <brief description of the skill>
    Aliases: <comma-separated alternative names>
    Skill Type: <category like Programming Language, Farming Technique, etc.>
    Skill Usages: <comma-separated key applications>
    
    Subsets:
    For dynamic subsets, first output a category name followed by a colon,
    then on subsequent lines list related items, each prefixed with "- ".Please don't add any extra symbols or stars in the beginning or at the end
    
    Example:
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
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    print(content)
    in_subsets_section = False
    current_section = None
    for line in content.split("\n"):
        line = line.strip()
        
        # Process main fields
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
        
        # Process dynamic subset categories only after "Subsets:" is reached
        elif in_subsets_section:
            # Detect a new category header (line ending with a colon)
            if line.endswith(":") and not line.startswith("-"):
                current_section = line[:-1].strip().lower().replace(" ", "_")
                extracted_data["subsets"][current_section] = []
            # Detect list items under the current category
            elif current_section and line.startswith("- "):
                extracted_data["subsets"][current_section].append(line[2:])
    
    return extracted_data

if __name__ == "__main__":
    skill = input("Enter a skill (e.g., 'Java'): ").strip()
    result = extract_related_skills(skill)
    print(json.dumps(result, indent=4))
