import os
import time
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

def extract_domain_skills(domain):
    web_pages = fetch_related_content(f"essential skills for {domain}")
    skills = set()
    
    for page in web_pages:
        print(f"Fetching data from: {page}")
        
        prompt = f"""
        Extract essential skills for a career in '{domain}' from this webpage: {page}
        
        Provide a concise bullet list of key skills with no redundant or extraneous information.
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

if __name__ == "__main__":
    domain = input("Enter a domain (e.g., 'agriculture'): ").strip()
    result = extract_domain_skills(domain)
    print(json.dumps(result, indent=4))
