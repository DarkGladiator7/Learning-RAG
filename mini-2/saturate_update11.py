import json
import os
import re
from subdomain_agent11 import extract_subdomain_details
from domain_agent11 import extract_domain_subfields
from langchain.chat_models import init_chat_model

# Initialize LLM for optimization
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

JSON_FILE = "master_it_subdomains.json"

def load_json_data():
    """Loads existing JSON data into a dictionary for efficient lookup."""
    if os.path.exists(JSON_FILE):
        print("Loading existing JSON data...")
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    print("No existing JSON data found, creating new structure.")
    return {}

def save_json_data(data):
    """Saves updated data back into the JSON file."""
    print("Saving updated subdomain data to JSON file...")
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("JSON file successfully updated.")

def optimize_with_llm(existing_entry, new_entry):
    """Uses LLM to intelligently merge and optimize data for a specific subdomain."""
    print(f"Optimizing subdomain: {existing_entry.get('Subdomain Name', 'Unknown')}...")

    prompt = f"""
    Optimize the IT subdomain dataset by merging these two entries while ensuring no redundant or repetitive information.

    Existing Entry:
    {json.dumps(existing_entry, indent=4)}

    New Data:
    {json.dumps(new_entry, indent=4)}

    Ensure:
    - No duplicate subdomain names
    - Do not add or combine on your own just check if any data is missing and add it
    - No redundant or repeated subdomain descriptions
    - Do not merge the subdomain name at any cost
    - Ensure technical skills in the existing entry are fully saturated and comprehensive
    - Only add missing and useful details
    - Keep subsets properly structured
    - Don't add any extra information or symbols in the output strictly follow the format
    - Use the same format as the existing entry
    
    Provide the final optimized JSON object.
    """
    
    print("Sending prompt to LLM for optimization...")
    response = llm.invoke(prompt)
    print("Received response from LLM.")
    raw_response = response.content.strip()
    match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    json_str = match.group(0)
    try:
        optimized_data = json.loads(json_str)
        print(f"Optimization successful for {existing_entry.get('Subdomain Name', 'Unknown')}.")
        return optimized_data
    except json.JSONDecodeError:
        print("Error! LLM returned an invalid JSON format. Using the existing best data.")
        return existing_entry  # Fallback to the existing best data

def infer_main_domain_with_llm(subdomain_name):
    """Uses LLM to find the correct main domain for a given subdomain."""
    print(f"Inferring main domain for '{subdomain_name}' using LLM...")

    prompt = f"""
    The user has provided a subdomain: "{subdomain_name}".

    Determine the most appropriate **main IT domain** under which this subdomain should fall.

    Ensure:
    - The domain is well-recognized in the tech industry.
    - Only return a **single domain name**.
    - The output must be a JSON string with the format: {{"domain": "correct_domain_name"}}.

    Example output:
    {{"domain": "Artificial Intelligence"}}
    """

    print("Sending prompt to LLM for domain inference...")
    response = llm.invoke(prompt)
    print(response.content.strip())
    try:
        inferred_domain = json.loads(response.content.strip())
        
        if "domain" in inferred_domain:
            print(f"LLM inferred main domain: {inferred_domain['domain']}")
            return inferred_domain["domain"]
    except json.JSONDecodeError:
        print("Error! LLM returned an invalid JSON format. Falling back to default.")

    return "General IT"  # Default fallback domain


def normalize_subdomain_with_llm(subdomain_name):
    """Uses LLM to normalize subdomain names dynamically."""
    print(f"Normalizing subdomain name '{subdomain_name}' using LLM...")

    prompt = f"""
    Given an IT subdomain in a similar name used for it, identify the most generalized name of the given input(subdomain). 

    Subdomain: "{subdomain_name}"

    Instructions:
    - Return a JSON object where "subdomain" is the given input, and you have to return the most generalized name of the given input.
    - Do not return any Python code.
    - Example if input is ml then output has to be:
    - Do not add any extra info just this output
    {{"subdomain": "Machine Learning"}}
    """


    print("Sending normalization request to LLM...")
    response = llm.invoke(prompt)
    print(response.content.strip())

    try:
        normalized_name = json.loads(response.content.strip())
        print(normalized_name)
        if "subdomain" in normalized_name:
            print(f"LLM normalized subdomain: {normalized_name['subdomain']}")
            return normalized_name["subdomain"]
    except json.JSONDecodeError:
        print("Error! LLM returned an invalid JSON format. Using original name.")

    return subdomain_name  # Default fallback

def update_or_add_subdomains(domain_query):
    """Ensures the correct main domain is identified before updating subdomains."""

    print("Starting update_or_add_subdomains function...")

    # Load existing JSON data
    existing_data = load_json_data()
    print("Loaded existing JSON data.")

    # Check if domain already exists in JSON
    if domain_query in existing_data:
        print(f"Domain '{domain_query}' found in existing data.")
        domain_name = domain_query
        subdomains = list(existing_data[domain_query].keys())  # Extract existing subdomains
    else:
        print(f"Domain '{domain_query}' not found. Extracting subdomains...")
        domain_result = extract_domain_subfields(domain_query)

        domain_name = domain_result.get("Domain")
        if not domain_name:
            print(f"Warning: Could not determine the main domain for '{domain_query}'. Using LLM to infer...")
            domain_name = infer_main_domain_with_llm(domain_query)

        subdomains = domain_result.get("Subdomains", []) or []
    
    # If no subdomains are found, infer one using LLM
    if not subdomains:
        print(f"Warning: No subdomains found for '{domain_name}'. Using LLM to infer one...")
        inferred_subdomain = normalize_subdomain_with_llm(domain_query)
        subdomains = [inferred_subdomain]

    print(f"Processing subdomains under '{domain_name}'...")

    updated_subdomains = []

    for subdomain in subdomains:
        # Normalize the subdomain name using LLM
        normalized_subdomain = normalize_subdomain_with_llm(subdomain)
        subdomain_key = normalized_subdomain.lower()

        print(f"\nProcessing subdomain: {normalized_subdomain}")

        # Ensure the subdomain is NOT the same as the domain itself
        if subdomain_key == domain_name.lower():
            print(f"Skipping redundant subdomain '{normalized_subdomain}' as it matches its main domain.")
            continue

        # Fetch new subdomain details
        new_subdomain_data = extract_subdomain_details(subdomain)

        # If extraction fails, create a minimal entry
        if not new_subdomain_data:
            print(f"Error: No details extracted for {normalized_subdomain}. Creating minimal entry.")
            new_subdomain_data = {
                "subdomain_name": normalized_subdomain,
                "subdomain_description": "No description available",
                "technical_skills": [],
                "soft_skills": [],
                "industry_specific_skills": [],
                "certifications": []
            }

        print(f"Extracted details for {normalized_subdomain}: {json.dumps(new_subdomain_data, indent=4)}")

        # Ensure the main domain exists in the JSON structure
        if domain_name not in existing_data:
            existing_data[domain_name] = {}

        # Check if the subdomain already exists
        if subdomain_key in existing_data[domain_name]:
            print(f"Updating existing subdomain: {normalized_subdomain}...")

            # Fetch current data and optimize using LLM
            current_data = existing_data[domain_name][subdomain_key]
            optimized_data = optimize_with_llm(current_data, new_subdomain_data)

            if not optimized_data:
                print(f"Error: Optimization failed for {normalized_subdomain}. Keeping old data.")
                continue

            print(f"Optimized data for {normalized_subdomain}: {json.dumps(optimized_data, indent=4)}")
            existing_data[domain_name][subdomain_key] = optimized_data
        else:
            print(f"Adding new subdomain: {normalized_subdomain} under '{domain_name}'...")
            existing_data[domain_name][subdomain_key] = new_subdomain_data

        updated_subdomains.append(normalized_subdomain)

    if not updated_subdomains:
        print("No updates were made. Exiting function.")
        return {"status": "No updates", "domain": domain_name, "updated_subdomains": []}

    print("\nSaving updated JSON data...")
    save_json_data(existing_data)
    print("JSON file successfully updated.")

    return {
        "status": "Success",
        "domain": domain_name,
        "updated_subdomains": updated_subdomains
    }



if __name__ == "__main__":
    user_input = input("Enter the IT domain to update or add subdomains: ").strip()
    print(f"User entered domain: {user_input}")
    update_or_add_subdomains(user_input)
