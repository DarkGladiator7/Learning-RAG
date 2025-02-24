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
    """Extracts structured details for a given IT subdomain, focusing on relevant skills."""
    
    web_pages = fetch_related_content(f"{subdomain} in IT industry")
    extracted_data = {
        "subdomain_name": subdomain.capitalize(),
        "technical_skills": [],
        "soft_skills": [],
        "industry_specific_skills": [],
        "certifications": []
    }

    prompt = f"""
    Extract a comprehensive set of skills required to work in the '{subdomain}' subdomain in IT.
    Provide the following details in the same format without extra symbols:
    
    Technical Skills: <comma-separated list of programming languages, tools, frameworks>
    Soft Skills: <comma-separated list of communication, teamwork, problem-solving skills>
    Industry-Specific Skills: <comma-separated list of domain-specific expertise, regulations>
    Certifications: <comma-separated list of relevant certifications or degrees>
    
    Example:
    Technical Skills: Python, TensorFlow, PyTorch, SQL, Kubernetes
    Soft Skills: Communication, Problem-Solving, Teamwork, Critical Thinking
    Industry-Specific Skills: Data Preprocessing, Neural Networks, Model Deployment
    Certifications: AWS Certified ML Specialist, Google TensorFlow Developer Certificate
    """

    response = llm.invoke(prompt)
    content = response.content.strip()
    print(content)

    for line in content.split("\n"):
        line = line.strip()
        
        if line.startswith("Technical Skills:"):
            extracted_data["technical_skills"] = [skill.strip() for skill in line.split(": ", 1)[1].split(",")]
        elif line.startswith("Soft Skills:"):
            extracted_data["soft_skills"] = [skill.strip() for skill in line.split(": ", 1)[1].split(",")]
        elif line.startswith("Industry-Specific Skills:"):
            extracted_data["industry_specific_skills"] = [skill.strip() for skill in line.split(": ", 1)[1].split(",")]
        elif line.startswith("Certifications:"):
            extracted_data["certifications"] = [cert.strip() for cert in line.split(": ", 1)[1].split(",")]
    
    return extracted_data

if __name__ == "__main__":
    subdomain = input("Enter a subdomain (e.g., 'Machine Learning'): ").strip()
    result = extract_subdomain_details(subdomain)
    print(json.dumps(result, indent=4))
