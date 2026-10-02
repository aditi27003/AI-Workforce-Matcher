"""
AI Workforce Matcher
====================

Matches employees to a project using semantic search (RAG):

1. Stores employee skill profiles in a Pinecone vector index
   (Pinecone embeds the text with the integrated `llama-text-embed-v2` model).
2. Searches the index with a plain-English project scope.
3. Sends the best-matching profiles to Mistral, which recommends who
   should be assigned and explains why.

Usage:
    pip install -r requirements.txt
    cp .env.example .env      # then paste your own API keys into .env
    python vector_database.py
    python vector_database.py "Set up CI/CD pipelines on AWS with Docker"
"""

import os
import sys
import time

from dotenv import load_dotenv
from mistralai.client import Mistral
from pinecone import Pinecone

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
load_dotenv()  # reads API keys from a local .env file (never commit it)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()

INDEX_NAME = "projectmanagementdata"
NAMESPACE = "workforcemanagement"
EMBED_MODEL = "llama-text-embed-v2"
MISTRAL_MODEL = "mistral-small-latest"
TOP_K = 2

DEFAULT_PROJECT_SCOPE = (
    "We need a developer to build a reactive web dashboard interface "
    "with modern styling."
)

# Sample workforce data: employee name -> technical skills
EMPLOYEES = {
    "Alice": "Python, SQL, Machine Learning",
    "Bob": "React, Javascript, UI/UX",
    "Charlie": "AWS, Docker, Kubernetes",
    "David": "Power BI, Excel, Tableau",
    "Emma": "Java, Spring Boot, PostgreSQL",
}


# --------------------------------------------------------------------------
# Step 1 - Vector index
# --------------------------------------------------------------------------
def get_index(pc: Pinecone):
    """Create the Pinecone index (with integrated embeddings) if needed and connect to it."""
    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": EMBED_MODEL,
                "field_map": {"text": "content"},  # field that gets embedded
            },
        )

    # Wait until the index is ready to accept data
    while not pc.describe_index(INDEX_NAME).status.ready:
        time.sleep(1)

    return pc.Index(INDEX_NAME)


# --------------------------------------------------------------------------
# Step 2 - Store employee profiles
# --------------------------------------------------------------------------
def upsert_employees(index, employees: dict) -> None:
    """Turn each employee into a text record and upsert it; Pinecone embeds it automatically."""
    records = [
        {"_id": name, "content": f"Employee {name} is qualified in: {skills}."}
        for name, skills in employees.items()
    ]
    print(f"Upserting {len(records)} employee records to Pinecone...")
    index.upsert_records(namespace=NAMESPACE, records=records)
    time.sleep(2)  # give the index a moment to make new records searchable


# --------------------------------------------------------------------------
# Step 3 - Semantic search
# --------------------------------------------------------------------------
def search_matches(index, project_scope: str, top_k: int = TOP_K) -> list[dict]:
    """Return the best-matching employees as dicts: name, profile text and similarity score."""
    response = index.search(
        namespace=NAMESPACE,
        query={"inputs": {"text": project_scope}, "top_k": top_k},
    )

    matches = []
    for hit in response.result.hits:
        text = (hit.fields or {}).get("content", "")
        if text:
            matches.append({"name": hit.id, "profile": text, "score": hit.score})
    return matches


def search_candidates(index, project_scope: str, top_k: int = TOP_K) -> list[str]:
    """Return the matching employee profiles as readable lines for the LLM prompt."""
    return [
        f"{m['profile']} (similarity: {m['score']:.3f})"
        for m in search_matches(index, project_scope, top_k)
    ]


# --------------------------------------------------------------------------
# Step 4 - LLM recommendation
# --------------------------------------------------------------------------
def recommend_assignment(mistral: Mistral, project_scope: str, candidates: list[str]) -> str:
    """Ask Mistral who should be assigned, using only the retrieved candidates."""
    context = "\n".join(f"- {c}" for c in candidates) or "- (no matching employees found)"

    prompt = f"""You are a project management assistant. Based on the following project scope
and retrieved internal workforce data, determine who should be assigned to this project
and justify your choice briefly.

Project scope: {project_scope}

Available candidate context:
{context}
"""

    response = mistral.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    if not MISTRAL_API_KEY or not PINECONE_API_KEY:
        sys.exit(
            "Missing API keys. Copy .env.example to .env and add your "
            "MISTRAL_API_KEY and PINECONE_API_KEY."
        )

    project_scope = " ".join(sys.argv[1:]) or DEFAULT_PROJECT_SCOPE

    pc = Pinecone(api_key=PINECONE_API_KEY)
    mistral = Mistral(api_key=MISTRAL_API_KEY)

    index = get_index(pc)
    upsert_employees(index, EMPLOYEES)

    print(f"\nProject scope: {project_scope}")
    candidates = search_candidates(index, project_scope)

    print("\nTop matching employees:")
    for c in candidates:
        print(f"  - {c}")

    print("\nAI recommendation:")
    print(recommend_assignment(mistral, project_scope, candidates))


if __name__ == "__main__":
    main()
