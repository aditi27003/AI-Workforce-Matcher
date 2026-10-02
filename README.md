# AI Workforce Matcher

**Find the right person for a project by describing the project in plain English.**

AI Workforce Matcher is a small Retrieval-Augmented Generation (RAG) project. It stores employee skill profiles in a **Pinecone** vector database, finds the employees whose skills best match a project description using **semantic search**, and asks **Mistral AI** to recommend who should be assigned and why.

🌐 **Project page:** https://aditi27003.github.io/AI-Workforce-Matcher/
📓 **Try it in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditi27003/AI-Workforce-Matcher/blob/main/vector_database.ipynb)

---

## How It Works

```
Employee skills ──► Pinecone (auto-embedded with llama-text-embed-v2)
                                  │
Project scope ──► Semantic search ┘──► Top matching employees
                                              │
                                              ▼
                                   Mistral AI (LLM) ──► Recommendation + reason
```

1. **Store** – Each employee becomes a text record, e.g. *"Employee Bob is qualified in: React, Javascript, UI/UX."* Pinecone's integrated embedding model turns it into a vector.
2. **Search** – The project scope is embedded the same way and compared by meaning, not keywords. "Reactive web dashboard with modern styling" matches Bob's *React* and *UI/UX* skills even though the words differ.
3. **Recommend** – The top matches are passed to Mistral with the project scope, and the model picks the best person and explains the choice.

---

## Example

**Project scope:**
> We need a developer to build a reactive web dashboard interface with modern styling.

**Top matches from Pinecone:** Bob (React, Javascript, UI/UX) and the next closest profile.

**Mistral's answer:** recommends Bob, because his React and UI/UX skills fit a modern, reactive dashboard.

*(Exact wording and scores vary from run to run.)*

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Pinecone | Vector database with integrated embeddings |
| `llama-text-embed-v2` | Embedding model (hosted by Pinecone) |
| Mistral AI (`mistral-small-latest`) | Recommendation and justification |
| python-dotenv | Loads API keys from a local `.env` file |

---

## Sample Data

| Employee | Skills |
|---|---|
| Alice | Python, SQL, Machine Learning |
| Bob | React, Javascript, UI/UX |
| Charlie | AWS, Docker, Kubernetes |
| David | Power BI, Excel, Tableau |
| Emma | Java, Spring Boot, PostgreSQL |

Edit the `EMPLOYEES` dictionary in `vector_database.py` to use your own team.

---

## Getting Started

### Option 1 – Google Colab (easiest)

1. Click **Open in Colab** above.
2. Open the 🔑 **Secrets** panel in Colab and add `MISTRAL_API_KEY` and `PINECONE_API_KEY`.
3. Run all cells.

### Option 2 – Run locally

```bash
git clone https://github.com/aditi27003/AI-Workforce-Matcher.git
cd AI-Workforce-Matcher
pip install -r requirements.txt
cp .env.example .env        # then paste your API keys into .env
python vector_database.py
```

Try your own project description:

```bash
python vector_database.py "Set up CI/CD pipelines on AWS with Docker"
```

### Getting API keys

- **Pinecone:** sign up at [pinecone.io](https://www.pinecone.io) → API Keys
- **Mistral:** sign up at [console.mistral.ai](https://console.mistral.ai) → API Keys

> 🔒 Keys live only in your `.env` file or Colab Secrets. `.env` is listed in `.gitignore`, so it is never uploaded.

---

## Project Structure

```
AI-Workforce-Matcher/
├── vector_database.py     # Main script
├── vector_database.ipynb  # Colab notebook version
├── index.html             # Project web page (GitHub Pages)
├── requirements.txt       # Python dependencies
├── .env.example           # Template for API keys
├── .gitignore
└── README.md
```

---

## Future Enhancements

- Load employees from a CSV, spreadsheet or HR system
- Add availability, experience level and location as search filters
- Recommend a full team for multi-skill projects
- Web interface for entering project scopes
- Use Pinecone reranking for more accurate matches

---

## Author

**Aditi Srivastava**
