import csv
import json
import os
from subdomain_agent1 import extract_subdomain_details
from domain_agent1 import extract_domain_subfields
from langchain.chat_models import init_chat_model

# Initialize LLM for optimization
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

CSV_FILE = "master_it_subdomains.csv"

def load_csv_data():
    """Loads existing CSV data into a dictionary for efficient lookup."""
    data = {}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subdomain_name = row["Subdomain Name"].strip().lower()
                data[subdomain_name] = row
    return data

def save_csv_data(data):
    """Saves updated data back into the CSV."""
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Subdomain Name", "Subdomain Description", "Related Roles", "Key Technologies", "Required Skills", "Subsets"])
        writer.writeheader()
        for row in data.values():
            writer.writerow(row)

def optimize_with_llm(existing_entry, new_entry):
    """Uses LLM to intelligently merge and optimize data."""
    
    prompt = f"""
    Optimize the IT subdomain dataset by merging these two entries while ensuring no redundant or repetitive information.

    Existing Entry:
    {json.dumps(existing_entry, indent=4)}

    New Data:
    {json.dumps(new_entry, indent=4)}

    Ensure:
    - No duplicate subdomain names or related roles
    - No redundant or repeated subdomain descriptions
    - Reason youself for every part of 
    - Every part of the json should be saturated such that no more info can be added to it 
    - Only add missing and useful details
    - Keep subsets properly structured

    Provide the final optimized JSON object.
    """
    
    response = llm.invoke(prompt)
    
    try:
        optimized_data = json.loads(response.content.strip())
        return optimized_data
    except json.JSONDecodeError:
        print("Error! LLM returned an invalid JSON format. Using the existing best data.")
        return existing_entry  # Fallback to the existing best data

def update_or_add_subdomains(domain_query):
    """Checks for existing data, optimizes, and updates the CSV."""
    
    existing_data = load_csv_data()
    
    # Step 1: Get subdomains related to the given domain
    domain_result = extract_domain_subfields(domain_query)
    domain_name = domain_result.get("Domain", "Unknown Domain")
    subdomains = domain_result.get("Subdomains", [])

    print(f"\nChecking and optimizing IT domain '{domain_name}' subdomains...")

    for subdomain in subdomains:
        subdomain_key = subdomain.lower()

        # Fetch new subdomain details
        new_subdomain_data = extract_subdomain_details(subdomain)

        if subdomain_key in existing_data:
            print(f"Optimizing existing subdomain: {subdomain}")
            current_data = existing_data[subdomain_key]

            # Optimize using LLM
            optimized_data = optimize_with_llm(current_data, new_subdomain_data)

            # Update the dataset
            existing_data[subdomain_key] = optimized_data

        else:
            print(f"Adding new optimized subdomain: {subdomain}")
            existing_data[subdomain_key] = {
                "Subdomain Name": new_subdomain_data.get("subdomain_name", ""),
                "Subdomain Description": new_subdomain_data.get("subdomain_description", ""),
                "Related Roles": json.dumps(new_subdomain_data.get("related_roles", [])),
                "Key Technologies": json.dumps(new_subdomain_data.get("key_technologies", [])),
                "Required Skills": json.dumps(new_subdomain_data.get("required_skills", [])),
                "Subsets": json.dumps(new_subdomain_data.get("subsets", {}))
            }

    # Save updated CSV
    save_csv_data(existing_data)
    print("\n✔ CSV file successfully optimized and updated!")

if __name__ == "__main__":
    user_input = input("Enter the IT domain to update or add subdomains: ").strip()
    update_or_add_subdomains(user_input)
