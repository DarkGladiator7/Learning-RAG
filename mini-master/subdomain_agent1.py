import os
import json
import time
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  

# Initialize LLM
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

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

def extract_subdomain_details(subdomain):
    """Extracts structured details for a given IT subdomain."""
    
    web_pages = fetch_related_content(f"{subdomain} in IT industry")
    extracted_data = {
        "subdomain_name": subdomain.capitalize(),
        "subdomain_description": "",
        "related_roles": [],
        "key_technologies": [],
        "required_skills": [],
        "subsets": {}
    }

    prompt = f"""
    Extract structured information about the '{subdomain}' subdomain in IT.
    Provide the following details in the same format without extra symbols:
    
    Subdomain Description: <brief overview>
    Related Roles: <comma-separated job roles related to this subdomain>
    Key Technologies: <comma-separated major technologies/tools used>
    Required Skills: <comma-separated essential skills a person needs>
    
    Subsets:
    For dynamic subsets, first output a category name followed by a colon,
    then on subsequent lines list related items, each prefixed with "- ".
    
    Example:
    Popular Frameworks:
    - TensorFlow
    - PyTorch
    Common Tools:
    - Jupyter Notebook
    - VS Code
    """

    response = llm.invoke(prompt)
    content = response.content.strip()
    print(content)

    in_subsets_section = False
    current_section = None

    for line in content.split("\n"):
        line = line.strip()

        # Process main fields
        if line.startswith("Subdomain Description:"):
            extracted_data["subdomain_description"] = line.split(": ", 1)[1]
        elif line.startswith("Related Roles:"):
            extracted_data["related_roles"] = [role.strip() for role in line.split(": ", 1)[1].split(",")]
        elif line.startswith("Key Technologies:"):
            extracted_data["key_technologies"] = [tech.strip() for tech in line.split(": ", 1)[1].split(",")]
        elif line.startswith("Required Skills:"):
            extracted_data["required_skills"] = [skill.strip() for skill in line.split(": ", 1)[1].split(",")]
        elif line.startswith("Subsets:"):
            in_subsets_section = True
            continue

        # Process dynamic subset categories
        elif in_subsets_section:
            if line.endswith(":") and not line.startswith("-"):
                current_section = line[:-1].strip().lower().replace(" ", "_")
                extracted_data["subsets"][current_section] = []
            elif current_section and line.startswith("- "):
                extracted_data["subsets"][current_section].append(line[2:])

    return extracted_data

if __name__ == "__main__":
    subdomain = input("Enter a subdomain (e.g., 'Machine Learning'): ").strip()
    result = extract_subdomain_details(subdomain)
    print(json.dumps(result, indent=4))