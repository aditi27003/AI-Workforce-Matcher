"""
AI Workforce Matcher - live web app (Streamlit)

Type a project description and get:
  * real similarity scores from Pinecone semantic search
  * a recommendation from Mistral AI

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (API keys go in the app's Secrets)
"""

import os

import streamlit as st
from mistralai.client import Mistral
from pinecone import Pinecone

from vector_database import (
    DEFAULT_PROJECT_SCOPE,
    EMPLOYEES,
    get_index,
    recommend_assignment,
    search_matches,
    upsert_employees,
)

MAX_SCOPE_CHARS = 500

st.set_page_config(page_title="AI Workforce Matcher", page_icon="🧭", layout="centered")


# --------------------------------------------------------------------------
# API keys: Streamlit Secrets (cloud) or environment / .env (local)
# --------------------------------------------------------------------------
def get_secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass  # no secrets file locally
    return os.getenv(name, "").strip()


MISTRAL_API_KEY = get_secret("MISTRAL_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")


@st.cache_resource(show_spinner="Connecting to Pinecone and loading employees...")
def load_clients():
    """Connect once per server, make sure the index exists and holds the employee data."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_index(pc)
    upsert_employees(index, EMPLOYEES)
    mistral = Mistral(api_key=MISTRAL_API_KEY)
    return index, mistral


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("🧭 AI Workforce Matcher")
st.write(
    "Describe a project and the app finds the best-fit employees by **meaning**, "
    "using Pinecone semantic search, then asks **Mistral AI** who should take it."
)

if not MISTRAL_API_KEY or not PINECONE_API_KEY:
    st.error(
        "API keys are missing. Add `MISTRAL_API_KEY` and `PINECONE_API_KEY` "
        "to the app's Secrets (or to a local `.env` file)."
    )
    st.stop()

with st.expander("👥 Employees in the database", expanded=False):
    st.table([{"Employee": n, "Skills": s} for n, s in EMPLOYEES.items()])

examples = [
    DEFAULT_PROJECT_SCOPE,
    "I need someone for a data analytics job building sales reports.",
    "Set up CI/CD pipelines and deploy containers on AWS.",
    "Build a machine learning model to predict customer churn.",
    "Develop a backend REST API with a relational database.",
]

with st.form("match"):
    choice = st.selectbox("Pick an example or write your own below", examples)
    scope = st.text_area(
        "Project description",
        value=choice,
        max_chars=MAX_SCOPE_CHARS,
        height=110,
    )
    top_k = st.slider("How many candidates to retrieve", 1, len(EMPLOYEES), 3)
    submitted = st.form_submit_button("Find the best match", type="primary")

if submitted:
    scope = scope.strip()
    if not scope:
        st.warning("Please enter a project description.")
        st.stop()

    try:
        index, mistral = load_clients()

        with st.spinner("Searching Pinecone..."):
            matches = search_matches(index, scope, top_k)

        st.subheader("Semantic search results")
        st.caption("Live similarity scores from Pinecone (higher = closer match).")
        if not matches:
            st.info("No matching employees found.")
        for i, m in enumerate(matches):
            skills = EMPLOYEES.get(m["name"], "")
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{'🥇 ' if i == 0 else ''}{m['name']}** — {skills}")
            col2.markdown(f"`{m['score']:.3f}`")
            st.progress(max(0.0, min(1.0, float(m["score"]))))

        with st.spinner("Asking Mistral AI for a recommendation..."):
            candidates = [f"{m['profile']} (similarity: {m['score']:.3f})" for m in matches]
            answer = recommend_assignment(mistral, scope, candidates)

        st.subheader("AI recommendation")
        st.success(answer)

    except Exception as e:  # show a friendly message instead of a stack trace
        st.error(f"Something went wrong while contacting Pinecone or Mistral: {e}")

st.divider()
st.caption(
    "Built by Aditi Srivastava · "
    "[Source on GitHub](https://github.com/aditi27003/AI-Workforce-Matcher) · "
    "[Project page](https://aditi27003.github.io/AI-Workforce-Matcher/)"
)
