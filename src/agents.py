import os
from typing import Literal
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.state import AgentState

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize retrieval engine (reusing project 3 components)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="../03-hybrid-search-rag/data/chroma_db", embedding_function=embeddings)
all_data = vector_store.get()
docs = all_data.get("documents", [])
tokenized_docs = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Pydantic schema for strict hallucination grading
class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def retrieve_node(state: AgentState) -> AgentState:
    print(f"\n[Agent: Retriever] Searching for: '{state['question']}'")
    query = state.get("rewritten_query") or state["question"]
    
    # 1. Dense search
    vector_results = vector_store.similarity_search(query, k=2)
    dense_texts = [doc.page_content for doc in vector_results]
    
    # 2. Sparse BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = bm25_scores.argsort()[::-1][:2]
    sparse_texts = [docs[i] for i in top_bm25_idx if i < len(docs)]
    
    # Combine candidates & rerank
    candidates = list(set(dense_texts + sparse_texts))
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    top_chunks = [item[0] for item in scored[:2]]
    state["documents"] = top_chunks
    state["loop_count"] = state.get("loop_count", 0) + 1
    print(f"[Agent: Retriever] Retrieved and reranked {len(top_chunks)} high-precision chunks.")
    return state

def generate_node(state: AgentState) -> AgentState:
    print("[Agent: Generator] Generating response via Groq...")
    context = "\n\n".join(state["documents"])
    prompt = (
        f"You are an expert enterprise AI assistant. Answer accurately using ONLY the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        f"Answer:"
    )
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400
    )
    state["generation"] = response.choices[0].message.content
    return state

def hallucination_grader_node(state: AgentState) -> AgentState:
    print("[Agent: Grader] Checking answer grounding against retrieved docs...")
    context = "\n\n".join(state["documents"])
    answer = state["generation"]
    
    prompt = (
        f"You are a strict evaluator assessing whether an LLM generation is grounded in the retrieved context.\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        f"Is the generated answer fully grounded and supported by the context? Answer with 'yes' or 'no'."
    )
    
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    result = completion.choices[0].message.content.lower()
    grade = "yes" if "yes" in result else "no"
    print(f"[Agent: Grader] Grounding check result: {grade}")
    state["hallucination_grade"] = grade
    return state

def query_rewriter_node(state: AgentState) -> AgentState:
    print("[Agent: Rewriter] Rewriting query for improved retrieval...")
    query = state["question"]
    prompt = f"Rewrite the following user query to be more precise and optimized for keyword/vector search. Return ONLY the rewritten query:\n\nQuery: {query}"
    
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    rewritten = completion.choices[0].message.content.strip()
    print(f"[Agent: Rewriter] Rewritten query: '{rewritten}'")
    state["rewritten_query"] = rewritten
    return state
