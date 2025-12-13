#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 20:03:12 2025

@author: tienn
"""

import numpy as np
from chatbot.utils import load_data
from sentence_transformers import SentenceTransformer, util

class CVGuardrailRouter:
    def __init__(self,
                 model,
                 routes):
        # 1. Load a lightweight, fast model optimized for semantic similarity
        self.model = model
        
        # 2. Define "Anchor Questions" - the prototypical questions for each valid topic
        # These act as the centroids for your intent clusters.
        self.routes = routes
        
        # 3. Pre-compute embeddings for all anchors (This happens once at startup)
        # We flatten the list to store them efficiently
        self.intent_map = []
        all_sentences = []
        
        for intent, examples in self.routes.items():
            for example in examples:
                self.intent_map.append(intent)
                all_sentences.append(example)
        
        # Encode all anchors into a matrix
        self.anchor_embeddings = self.model.encode(all_sentences, convert_to_tensor=True)
        print("Guardrail Router initialized.")

    def route_query(self, user_query, threshold=0.35):
        """
        Takes a user query and returns the matching INTENT or None if blocked.
        """
        # 1. Encode the user's query
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        
        # 2. Calculate cosine similarity against all anchor questions
        cosine_scores = util.cos_sim(query_embedding, self.anchor_embeddings)[0]
        
        # 3. Find the best match
        best_match_idx = np.argmax(cosine_scores.cpu().numpy())
        best_score = cosine_scores[best_match_idx].item()
        best_intent = self.intent_map[best_match_idx]
        
        print(f"DEBUG: Query='{user_query}' | Best Match='{self.routes[best_intent][0]}' | Score={best_score:.4f}")

        # 4. The Guardrail Check
        if best_score < threshold:
            return {
                "allowed": False, 
                "reason": "out_of_scope", 
                "score": best_score
            }
        
        return {
            "allowed": True, 
            "intent": best_intent, 
            "score": best_score
        }

# --- usage Example ---
def main():
    # Initialize the router (Do this once when app starts)
    router = CVGuardrailRouter(model=SentenceTransformer('all-MiniLM-L6-v2'), routes=load_data('anchors.json'))

    # Simulate User Queries
    test_queries = [
        "How many years of experience you have with Python?",      # Should match 'skills'
        "How many time do you drink water a day?",           # Should be blocked
        "Tell me about your time at Google.",       # Should match 'experience'
        "Write me a poem about cats.",              # Should be blocked
        "How do I email you?"                       # Should match 'contact'
    ]

    print("\n--- Routing Results ---")
    for query in test_queries:
        result = router.route_query(query)
        
        if result["allowed"]:
            print(f"✅ PASSED: '{query}' -> Intent: [{result['intent'].upper()}]")
            # HERE: You would now fetch the specific CV text for this intent
            # and send it to Llama.
        else:
            print(f"⛔ BLOCKED: '{query}' (Score too low)")

if __name__ == "__main__":
    main()
