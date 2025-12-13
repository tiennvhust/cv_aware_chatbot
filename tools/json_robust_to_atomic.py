#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:20:51 2025

@author: tienn
"""

import json
import uuid
import argparse

def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_context_string(category, role, name, start, end):
    """Creates the human-readable string for the LLM"""
    if category == 'project':
        return f"In {category} \"{name}\", as the {role.lower()} ({start} to {end})"
    elif category == 'education':
        return f"During my {role.lower()} studies at {name} ({start} to {end})"
    else:
        return f"During my time as {role.lower()} at {name} ({start} to {end})"

def convert_robust_to_atomic(cv_data):
    atomic_list = []

    # --- HELPER: Process generic job/education blocks ---
    def process_block(block, category):
        # Normalize fields based on category (Education vs Experience)
        if category == 'education':
            name = block.get('school', 'Unknown School')
            role = f"{block.get('level', '')} in {block.get('field', '')}".strip()
        elif category == 'project':
            name = block.get('name', 'Unknown Project')
            role = block.get('role', 'Contributor')
        else:
            name = block.get('company', 'Unknown Company')
            role = block.get('title', 'Unknown Role')

        start_date = block.get('from', 'Unknown')
        end_date = block.get('to', 'Present')
        
        # Check if 'items' exists; if not, create a generic one from the block itself
        items = block.get('items', [])
        if not items:
            # Fallback if there are no bullet points (e.g., just a degree listing)
            items = [{
                "details": f"Completed {role} at {name}.", 
                "skills": []
            }]

        for item in items:
            chunk_id = f"{category[:3]}_{name[:3].lower().replace(' ', '')}_{str(uuid.uuid4())[:6]}"
            
            # Combine details for search
            text_content = item.get('details', '')
            
            # Generate the context string
            context_str = generate_context_string(category, role, name, start_date, end_date)
            
            atomic_entry = {
                "id": chunk_id,
                "type": category, # Useful for filtering later
                "role": role,
                "name": name,
                "start_date": start_date, # Kept raw for Math Calculator
                "end_date": end_date,     # Kept raw for Math Calculator
                "text": text_content,
                "skills": [s.lower() for s in item.get('skills', [])],
                "context_str": context_str
            }
            
            atomic_list.append(atomic_entry)

    # --- 1. Process Experience ---
    # It's a top-level list in your new JSON
    for exp in cv_data.get('experience', []):
        process_block(exp, 'experience')

    # --- 2. Process Education ---
    # It's nested inside 'profile'
    profile = cv_data.get('profile', {})
    for edu in profile.get('education', []):
        process_block(edu, 'education')

    # --- 3. Process Projects ---
    projects = cv_data.get('projects', [])
    for proj in projects:
        process_block(proj, 'project')

    return atomic_list

def main(cv_path : str):
    cv_source_data = load_data(cv_path)
    # --- Execution ---
    atomic_db = convert_robust_to_atomic(cv_source_data)

    # Save to your "Database" file
    with open('cv_atomic_db.json', 'w') as f:
        json.dump(atomic_db, f, indent=2)

    # Verify Output
    print(f"Successfully converted {len(atomic_db)} items.")
    print("\n--- Example Atomic Record ---")
    print(json.dumps(atomic_db[0], indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()
    main(args.input)