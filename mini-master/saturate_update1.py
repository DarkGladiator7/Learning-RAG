import csv
import json
import os
from skill_agent import extract_related_skills
from domain_agent import extract_domain_skills
from langchain.chat_models import init_chat_model

# Initialize LLM for optimization
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

CSV_FILE = "master_it_skills.csv"

def load_csv_data():
    """Loads existing CSV data into a dictionary for efficient lookup."""
    data = {}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                skill_name = row["Skill Name"].strip().lower()
                data[skill_name] = row
    return data

def save_csv_data(data):
    """Saves updated data back into the CSV."""
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Skill Name", "Skill Description", "Aliases", "Skill Type", "Skill Usages", "Subsets"])
        writer.writeheader()
        for row in data.values():
            writer.writerow(row)

def optimize_with_llm(existing_entry, new_entry):
    """Uses LLM to intelligently merge and optimize data."""
    
    prompt = f"""
    Optimize the IT skill dataset by merging these two entries while ensuring no redundant or repetitive information.

    Existing Entry:
    {json.dumps(existing_entry, indent=4)}

    New Data:
    {json.dumps(new_entry, indent=4)}

    Ensure:
    - No duplicate skill names or aliases (e.g., "Java" and "Java Developer" should be merged)
    - No redundant or repeated skill descriptions
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

def update_or_add_skills(domain_query):
    """Checks for existing data, optimizes, and updates the CSV."""
    
    existing_data = load_csv_data()
    
    # Step 1: Get domain-related skills
    domain_result = extract_domain_skills(domain_query)
    domain_name = domain_result.get("Domain", "Unknown Domain")
    required_skills = domain_result.get("Required Skills", [])
    
    print(f"\nChecking and optimizing IT domain '{domain_name}' skills...")

    for skill in required_skills:
        skill_key = skill.lower()

        # Fetch new skill details
        new_skill_data = extract_related_skills(skill)

        if skill_key in existing_data:
            print(f"Optimizing existing skill: {skill}")
            current_data = existing_data[skill_key]

            # Optimize using LLM
            optimized_data = optimize_with_llm(current_data, new_skill_data)

            # Update the dataset
            existing_data[skill_key] = optimized_data

        else:
            print(f"Adding new optimized skill: {skill}")
            existing_data[skill_key] = {
                "Skill Name": new_skill_data.get("skill_name", ""),
                "Skill Description": new_skill_data.get("skill_description", ""),
                "Aliases": json.dumps(new_skill_data.get("aliases", [])),
                "Skill Type": new_skill_data.get("skill_type", ""),
                "Skill Usages": json.dumps(new_skill_data.get("skill_usages", [])),
                "Subsets": json.dumps(new_skill_data.get("subsets", {}))
            }

    # Save updated CSV
    save_csv_data(existing_data)
    print("\n✔ CSV file successfully optimized and updated!")

if __name__ == "__main__":
    user_input = input("Enter the IT domain to update or add skills: ").strip()
    update_or_add_skills(user_input)
