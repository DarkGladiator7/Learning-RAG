import csv
import json
from domain_agent import extract_domain_skills
from skill_agent import extract_related_skills
import os
from dotenv import load_dotenv
load_dotenv()  

os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("GROQ_API_KEY")
os.getenv("NOMIC_API_KEY")
 
def main1():
    # User provides a domain query, for example: "essential skills for agriculture"
    domain_query = input("Enter your domain query (e.g., 'essential skills for agriculture'): ").strip()
    
    # Step 1: Get domain skills
    domain_result = extract_domain_skills(domain_query)
    domain_name = domain_result.get("Domain", "Unknown Domain")
    required_skills = domain_result.get("Required Skills", [])
    
    # Step 2: For each required skill, get detailed info using the skill agent
    detailed_skills = []
    i=0
    for skill in required_skills:
        print(f"Processing detailed info for skill: {skill}")
        detailed_info = extract_related_skills(skill)
        detailed_skills.append(detailed_info)
        i+=1
        if(i>6):
            break
    
    # Aggregate the final output
    final_output = {
        "Domain": domain_name,
        "Domain Skills": required_skills,
        "Detailed Skills": detailed_skills
    }
    
    # Print the final JSON output
    print("Final JSON Output:")
    print(json.dumps(final_output, indent=4))
    
    # Save detailed skills info to CSV
    csv_file = "final_output.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Skill Name", "Skill Description", "Aliases", "Skill Type", "Skill Usages", "Subsets"]
        writer.writerow(header)
        for skill in detailed_skills:
            aliases = json.dumps(skill.get("aliases", []))
            skill_usages = json.dumps(skill.get("skill_usages", []))
            subsets = json.dumps(skill.get("subsets", {}))
            row = [
                skill.get("skill_name", ""),
                skill.get("skill_description", ""),
                aliases,
                skill.get("skill_type", ""),
                skill_usages,
                subsets
            ]
            writer.writerow(row)
    
    print(f"Final output stored in {csv_file}")

if __name__ == "__main__":
    main1()
