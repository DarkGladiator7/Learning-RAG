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

def extract_domain_subfields(domain):
    """Extracts major subfields for a given IT domain using LLM and web content."""

    web_pages = fetch_related_content(f"subfields of {domain} in IT")
    subfields = set()

    for page in web_pages:
        print(f"Fetching data from: {page}")

        prompt = f"""
        Identify and list all major subfields within the '{domain}' domain in IT based on this webpage: {page}.
        
        Provide a concise list with the below format and don't use any extra info or erroneous stars with the output and with no redundant or extraneous information.
        
        Format:
        - Subfield1
        - Subfield2
        - Subfield3
        """

        response = llm.invoke(prompt)
        content = response.content.strip()
       

        for line in content.split("\n"):
            if line.startswith("- "):
                subfields.add(line[2:])  # Extract subfield name

    return {"Domain": domain, "Subfields": list(subfields)}


if __name__ == "__main__":
    domain = input("Enter a domain (e.g., 'agriculture'): ").strip()
    result = extract_domain_subfields(domain)
    print(json.dumps(result, indent=4))
