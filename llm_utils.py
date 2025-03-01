import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def load_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY is missing. Check your .env file!")
        return None
    return api_key

def initialize_llm(api_key):
    llm_name = "gpt-3.5-turbo"
    return ChatOpenAI(model=llm_name)
