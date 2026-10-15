# 04-LangGraph-Agentic-RAG

An advanced, stateful, multi-agent Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Groq**, **Pydantic**, **ChromaDB**, **HuggingFace**, and **BM25 Hybrid Search**. This architecture transitions from rigid linear pipelines to a **cyclic state machine**, allowing specialized AI agents to collaborate, evaluate grounding, and autonomously self-correct when hallucinations or missing information are detected.

---

## Architecture

```
[User Query] 
     │
     ▼
[Retrieve Node] ──> (BM25 + Dense Vector + Cross-Encoder Reranker)
     │
     ▼
[Generate Node] ──> (Groq LPU Acceleration: openai/gpt-oss-20b)
     │
     ▼
[Hallucination Grader Node] ──> (Pydantic Structured Output)
     │
     ├──> [If Grounded ("yes")] ──> End / Return Final Answer
     │
     └──> [If Ungrounded ("no") & Loops <= 2] ──> [Query Rewriter Node] ──> Loop back to [Retrieve Node]
```

---

## Agent Nodes & State Flow

1. **Retriever Node (`retrieve_node`):** Executes hybrid sparse (BM25) and dense (ChromaDB + HuggingFace) search, followed by a `ms-marco-MiniLM-L-6-v2` cross-encoder reranking step to extract top-precision context chunks.
2. **Generator Node (`generate_node`):** Uses Groq's high-performance LLM engine to synthesize a grounded response strictly based on the retrieved context.
3. **Hallucination Grader (`hallucination_grader_node`):** Employs strict Pydantic schema validation (`GradeHallucination`) to evaluate whether the generated response is fully supported by the retrieved source documents.
4. **Query Rewriter (`query_rewriter_node`):** Refines and re-optimizes vague or ungrounded queries when the self-correction loop triggers a retry.

---

## Project Structure

```text
04-langgraph-agentic-rag/
│
├── data/                      # Shared or local database linkage
├── src/
│   ├── __init__.py
│   ├── state.py               # TypedDict schema defining graph state
│   ├── agents.py              # Specialized agent nodes & grading logic
│   └── graph.py               # LangGraph state machine & conditional routing
├── tests/
│   └── test_agent.py          # End-to-end agentic test execution script
├── .env                       # Environment variables (Groq API Key)
├── requirements.txt           # Project dependencies
└── README.md
```

---

## Tech Stack

* **Language:** Python 3.10+
* **Orchestration & State Machines:** `langgraph`, `langchain-core`
* **Data Validation:** `pydantic` (Structured Outputs)
* **LLM Engine:** Groq API (`openai/gpt-oss-20b`)
* **Retrieval & Reranking:** ChromaDB, HuggingFace Embeddings, Okapi BM25, Sentence-Transformers Cross-Encoder

---

## Quickstart Instructions

### 1. Clone & Set Up Virtual Environment
```powershell
git clone https://github.com/drazaH21/04-langgraph-agentic-rag.git
cd 04-langgraph-agentic-rag
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. Run the Agentic Test Suite
Execute the multi-agent graph test script to observe autonomous retrieval, generation, and grading in real-time:
```powershell
python -m tests.test_agent
```

---

## Evaluation & Execution Log

| Test Query | Initial Retrieval | Grader Assessment | Route Decision | Final Output Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **"How do Iron Mountain data pipelines run?"** | Retrieved top-2 high-precision chunks | **`yes`** (Fully grounded) | Finish Workflow (1 Loop) | Successfully extracted Python, BigQuery, Looker, and daily 02:00 UTC schedule with zero hallucinations. |

---

## License
Distributed under the MIT License. See `LICENSE` for more information.