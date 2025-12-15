#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 20:07:49 2025

@author: tienn
"""

import json
import numpy as np
import argparse
from chatbot.utils import load_data, match_skills
from sentence_transformers import SentenceTransformer, util

class CVRetrievalEngine:
    def __init__(self,
                 model,
                 corpus):
        self.model = model
        
        # Load the Atomic Data
        self.corpus = corpus
            
        # 1. Pre-compute Embeddings for all atomic chunks
        # We embed the 'text' field (description + details)
        corpus_texts = [doc['text'] for doc in self.corpus]
        self.corpus_embeddings = self.model.encode(corpus_texts, convert_to_tensor=True)
        
        # 2. Build a "Skill Index" for fast filtering
        # Set of all unique skills in your CV for quick lookup
        self.all_known_skills = set()
        for doc in self.corpus:
            for skill in doc['skills']:
                self.all_known_skills.add(skill.lower())
                
        print(f"Engine ready. Loaded {len(self.corpus)} facts and {len(self.all_known_skills)} unique skills.")

    def intent_matching(self, user_query, user_intent):
        print(f"🔍 [Search] Intent: {user_intent.capitalize()}")
        candidate_indices = []
        if user_intent == "skills":
            detected_skills = match_skills(self.all_known_skills, user_query.lower())
            if detected_skills:
                print(f"   [Filter] Detected specific skills: {detected_skills}")
                # Only keep documents that contain AT LEAST ONE of the detected skills
                for idx, doc in enumerate(self.corpus):
                    # Check intersection between doc skills and detected skills
                    if any(s in doc['skills'] for s in detected_skills):
                        candidate_indices.append(idx)
            else:
                print("   [Filter] No specific skills detected in query. Using full corpus.")
                candidate_indices = list(range(len(self.corpus)))
        elif user_intent in ["experience", "education", "projects"]:
            candidate_indices = [idx for idx, doc in enumerate(self.corpus) if doc['type'] == user_intent]
        elif user_intent in ["contact"]:
            candidate_indices = []
        else:
            candidate_indices = list(range(len(self.corpus)))
        return candidate_indices

    def search(self, user_query, user_intent, top_k=3):
        """
        Hybrid Search:
        1. Identify skills in query -> Filter corpus (Indices).
        2. Semantic Search on filtered indices -> Rank results.
        """
        # --- Step 1: Intent Matching & Filtering ---
        candidate_indices = self.intent_matching(user_query=user_query, user_intent=user_intent)

        if not candidate_indices:
            return []

        # --- Step 2: Semantic Search on Candidates ---
        # We only compare the query against the embeddings of the CANDIDATES
        
        # Encode the query
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        
        # Filter the pre-computed embeddings
        target_embeddings = self.corpus_embeddings[candidate_indices]
        
        # Calculate cosine similarity
        hits = util.semantic_search(query_embedding, target_embeddings, top_k=top_k)[0]
        
        # --- Step 3: Format Results ---
        results = []
        for hit in hits:
            # The 'corpus_id' in 'hit' corresponds to the index in 'target_embeddings'
            # We need to map it back to the original 'self.corpus' index
            original_index = candidate_indices[hit['corpus_id']]
            score = hit['score']
            doc = self.corpus[original_index]
            
            results.append({
                "score": score,
                "context": doc['context_str'],
                "content": doc['text'],
                "skills": doc['skills']
            })
            
        return results

# --- Usage Example ---
def main(db_path : str):
    engine = CVRetrievalEngine(SentenceTransformer('all-MiniLM-L6-v2'), db=load_data('cv_atomic_db.json'))
    
    # Test 1: Specific Skill Query
    query1 = "How are you familiar with Python?"
    print(f"\nQuery: '{query1}'")
    matches = engine.search(query1)
    for m in matches:
        print(f" - [{m['score']:.2f}] {m['context']}: {m['content']}")
    
    # Test 2: General Leadership Query (No hard skill defined in query usually, but 'leadership' is a skill in our DB)
    query2 = "Tell me about your leadership experience."
    print(f"\nQuery: '{query2}'")
    matches = engine.search(query2)
    for m in matches:
        print(f" - [{m['score']:.2f}] {m['context']}: {m['content']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default='cv_atomic_db.json')
    args = parser.parse_args()
    main(args.input_path)