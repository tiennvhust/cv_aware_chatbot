#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:30:40 2025

@author: tienn
"""
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chatbot.orchestrator import CVOrchestrator
from tools.pickle_data import SecureDataTool
from typing import Dict, Any

anchors_path = "data/anchors.pkl"
data_path = "data/cv_atomic_db.pkl"
contacts_path = "data/contacts.pkl"

def load_data(anchors_path:str, data_path:str, contacts_path:str) -> Dict[str,Any]:
    if "encode_key" in st.secrets:
        data_tool = SecureDataTool(st.secrets["encode_key"])
    else:
        raise FileNotFoundError("Could not load encryption key!")
    
    anchors = data_tool.load_encrypted_pickle(anchors_path)
    data = data_tool.load_encrypted_pickle(data_path)
    contacts = data_tool.load_encrypted_pickle(contacts_path)
    
    return {"anchors" : anchors,
            "database" : data,
            "contacts" : contacts}

def main():
    if "api_key" in st.secrets:
        API_KEY = st.secrets["api_key"]
    else:
        raise FileNotFoundError("Could not load API key!")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=API_KEY
    )
    
    inputs = load_data(anchors_path, data_path, contacts_path)

    cv_filter = CVOrchestrator(**inputs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_promt}"),
        ("user", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()

    # STREAMLIT UI
    st.title("Candidate Profile Assistant")
    st.markdown("Ask me about my skills, experience, or suitability for a role.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Handle user input
    if user_input := st.chat_input("Ask a question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        results = cv_filter.handle_query(user_input)
        if results['status'] == "success":
            response = chain.invoke({
                "system_promt" : results['system_prompt'],
                "question" : results['user_query']
            })
        else:
            response = results['response']
    
        # Add AI response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        
if __name__ == "__main__":
    main()