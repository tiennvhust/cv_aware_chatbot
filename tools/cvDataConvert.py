#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 20:06:43 2025

@author: tienn
"""

import json
import uuid
import argparse

def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_nested_to_atomic(nested_data):
    """
    Flattens hierarchical CV data into atomic 'facts' optimized for RAG.
    """
    atomic_list = []
    
    for job in nested_data:
        # Parent Context (The "Where" and "When")
        company = job.get('company', 'Unknown')
        role = job.get('role', 'Unknown')
        date_range = job.get('date', 'Unknown')
        
        for item in job.get('highlights', []):
            # Create a unique ID for this specific bullet point
            chunk_id = f"{company[:3].lower()}_{str(uuid.uuid4())[:8]}"
            
            # Combine 'description' and 'details' for the full searchable text
            full_text = f"{item['description']} {item.get('details', '')}"
            
            # Create the Context String (Crucial for the LLM)
            # This ensures the LLM knows the timeframe even if it only sees this one chunk.
            context_str = f"During my time as {role} at {company} ({date_range})"
            
            atomic_entry = {
                "id": chunk_id,
                "role": role,
                "organization": company,
                "date": date_range,
                "text": full_text,
                "skills": [s.lower() for s in item.get('skills', [])], # Normalize to lowercase
                "context_str": context_str
            }
            
            atomic_list.append(atomic_entry)
            
    return atomic_list

def main(nested_cv_path : str):
    nested_cv_data = load_data(nested_cv_path)
    # Run conversion
    atomic_cv = convert_nested_to_atomic(nested_cv_data)
    
    # Save to file (This is your 'Database')
    with open('cv_atomic_db.json', 'w') as f:
        json.dump(atomic_cv, f, indent=2)
    
    print(f"Converted {len(atomic_cv)} atomic records.")
    print(json.dumps(atomic_cv[0], indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()
    main(args.input)