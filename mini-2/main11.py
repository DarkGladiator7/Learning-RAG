import csv
import json
from domain_agent11 import extract_domain_subfields
from subdomain_agent11 import extract_subdomain_details
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  

os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("GROQ_API_KEY")
os.getenv("NOMIC_API_KEY")

def main():
    # User provides a domain (e.g., "Artificial Intelligence")
    domain_query = input("Enter a domain (e.g., 'Artificial Intelligence'): ").strip()
    
    # Step 1: Get subdomains under the given domain
    domain_result = extract_domain_subfields(domain_query)
    print(domain_result)
    domain_name = domain_result.get("Domain", "Unknown Domain")
    subdomains = domain_result.get("Subfields", [])

    if not subdomains:
        print(f"No subdomains found for {domain_name}. Exiting...")
        return

    # Step 2: Extract details for each subdomain
    detailed_subdomains = []
    i = 0
    for subdomain in subdomains:
        print(f"Processing details for subdomain: {subdomain}")
        subdomain_info = extract_subdomain_details(subdomain)
        detailed_subdomains.append(subdomain_info)
        
        i += 1
        if i > 6:  # Limit the number of API calls
            break
    
    # Aggregate the final output
    final_output = {
        "Domain": domain_name,
        "Subdomains": subdomains,
        "Detailed Subdomains": detailed_subdomains
    }

    # Print the final JSON output
    print("Final JSON Output:")
    print(json.dumps(final_output, indent=4))

    '''# Save subdomain details to CSV
    csv_file = "final_subdomains_output.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Subdomain Name", "Subdomain Description", "Related Roles", "Key Technologies", "Required Skills", "Subsets"]
        writer.writerow(header)
        
        for subdomain in detailed_subdomains:
            related_roles = json.dumps(subdomain.get("related_roles", []))
            key_technologies = json.dumps(subdomain.get("key_technologies", []))
            required_skills = json.dumps(subdomain.get("required_skills", []))
            subsets = json.dumps(subdomain.get("subsets", {}))

            row = [
                subdomain.get("subdomain_name", ""),
                subdomain.get("subdomain_description", ""),
                related_roles,
                key_technologies,
                required_skills,
                subsets
            ]
            writer.writerow(row)

    print(f"Final output stored in {csv_file}")'''
    json_file = "final_subdomains_output.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    
    print(f"Final output stored in {json_file}")

if __name__ == "__main__":
    main()