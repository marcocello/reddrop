import streamlit as st
import re
from typing import List
import nltk
from collections import Counter

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'content' not in st.session_state:
        st.session_state.content = ""
    if 'goal' not in st.session_state:
        st.session_state.goal = "conversational"
    if 'session_id' not in st.session_state:
        st.session_state.session_id = ""
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []
    if 'subreddits' not in st.session_state:
        st.session_state.subreddits = []
    if 'recommended_threads' not in st.session_state:
        st.session_state.recommended_threads = []
    if 'selected_threads' not in st.session_state:
        st.session_state.selected_threads = []
    if 'draft_comments' not in st.session_state:
        st.session_state.draft_comments = []
    if 'dataframe_selection' not in st.session_state:
        st.session_state.dataframe_selection = {"rows": [], "columns": []}