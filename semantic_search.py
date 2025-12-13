#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:44:26 2025

@author: tienn
"""

from sentence_transformers import SentenceTransformer, util
import json

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load work history
with open("work_history.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten experience into searchable text
texts = []
meta = []
for block in data["experience"]:
    company = block["company"]
    for item in block["items"]:
        combined = item["details"] + " " + " ".join(item["skills"])
        texts.append(combined)
        meta.append({
            "title": item["title"],
            "company": company,
            "details": item["details"],
            "skills": item["skills"]
        })

# Embed all entries
embeddings = model.encode(texts, convert_to_tensor=True)

def search_skills(query: str, top_k: int = 5):
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=top_k)[0]
    results = []
    for hit in hits:
        i = hit['corpus_id']
        score = hit['score']
        entry = meta[i]
        results.append({
            "score": round(score, 3),
            "title": entry["title"],
            "company": entry["company"],
            "details": entry["details"],
            "skills": entry["skills"]
        })
    return results