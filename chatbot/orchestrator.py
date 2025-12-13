#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 21:58:13 2025

@author: tienn
"""
from chatbot.guardrailRouter import CVGuardrailRouter
from chatbot.retrievalEngine import CVRetrievalEngine
from chatbot.utils import total_experience_years, compute_skill_experience, format_atomic_data, format_years
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any, Tuple

class CVOrchestrator:
    def __init__(self, anchors, database, contacts):
        self.cv_data = database # The "Database" for Math (Step 3)
        self.anchors = anchors
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.router = CVGuardrailRouter(self.embedding_model, self.anchors) # The "Bouncer" (Step 1)
        self.retriever = CVRetrievalEngine(self.embedding_model, self.cv_data) # The "Librarian" (Step 2)
        formated_data = format_atomic_data(self.cv_data)
        self.skills_data = compute_skill_experience(formated_data)
        self.experience_data = [data for data in formated_data if data['type'] == "experience"]
        self.email_add = contacts["email_add"]
        self.phone_num = contacts["phone_num"]
        
    def fact_eject(self, intent:str, user_query:str) -> List[str]:
        facts = []
        if intent == "skills":
            # Re-use the retriever's skill set to find keywords
            detected_skills = [s for s in self.retriever.all_known_skills if s in user_query.lower()]
            
            for skill_name in detected_skills:
                facts.append(f"- Total experience with {skill_name}: {format_years(self.skills_data.get(skill_name)['months'])} years.")
        elif intent == "experience":
            facts.append(f"- Total years of professional experience: {total_experience_years(self.experience_data)} years.")
        elif intent == "contact":
            facts.append(f"- Email address: {self.email_add}.")
            facts.append(f"- Phone number: {self.phone_num}.")
        return facts

    def handle_query(self, user_query):
        """
        Full pipeline: Guardrail -> Skill Check -> Retrieval -> Prompt
        """
        
        # --- PHASE 1: GUARDRAIL (The Router) ---
        route_result = self.router.route_query(user_query)
        
        # If the router says "Block", we stop immediately.
        # This saves API costs and prevents jailbreaks.
        if not route_result['allowed']:
            return {
                "status": "blocked",
                "response": "I can only answer questions about my professional profile, skills, and work experience."
            }

        print(f"✅ Intent Allowed: {route_result['intent']}")

        # --- PHASE 2: SKILL CALCULATION (The Fact Injector) ---
        # We detect skills regardless of the intent (unless it's purely 'contact')
        quantitative_facts = self.fact_eject(route_result['intent'], user_query)

        # --- PHASE 3: RETRIEVAL (The Semantic Search) ---
        # We fetch text chunks based on the query
        search_results = self.retriever.search(user_query)
        
        # --- PHASE 4: PROMPT ASSEMBLY ---
        # If no results found and no skills detected, we might need a fallback
        if not search_results and not quantitative_facts:
             return {
                "status": "no_data",
                "response": "I couldn't find specific details about that in the CV, but I can tell you about my general background."
            }

        # Format the context
        context_text = "\n".join([f"- [{r['context']}]: {r['content']}" for r in search_results])
        facts_text = "\n".join(quantitative_facts) if quantitative_facts else "No specific quantitative data for this query."
        
        # Final System Prompt
        system_prompt = f"""
        You are an AI assistant representing a candidate. Answer the user's question based ONLY on the context below.
        
        === USER INTENT ===
        The user is asking about: {route_result['intent'].upper()}
        
        === KEY FACTS (Immutable Numbers) ===
        {facts_text}
        
        === RELEVANT EXPERIENCE SNIPPETS ===
        {context_text}
        
        === INSTRUCTIONS ===
        1. If 'Key Facts' are present, cite the number of years explicitly.
        2. Use the 'Experience Snippets' to provide evidence and examples.
        3. Keep the tone professional, confident, and concise.
        """

        return {
            "status": "success",
            "system_prompt": system_prompt,
            "user_query": user_query
        }

# --- USAGE EXAMPLE ---

# 1. Setup (Run once at startup)
# router = CVGuardrailRouter()
# engine = CVRetrievalEngine()
# bot = CVOrchestrator()

# 2. Runtime
# query = "Do you know Python?"
# result = bot.handle_query(query)

# if result['status'] == 'success':
#     # Send to LLM
#     llm_response = call_llama_api(result['system_prompt'], result['user_query'])
#     print(llm_response)
# else:
#     # Return blocked message directly
#     print(result['response'])