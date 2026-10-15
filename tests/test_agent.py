from src.graph import app

def test_multi_agent_rag():
    print("=== Initializing Stateful Multi-Agent RAG Test ===")
    
    initial_state = {
        "question": "How do Iron Mountain data pipelines run?",
        "rewritten_query": "",
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "hallucination_grade": None
    }

    print(f"\nRunning graph for query: '{initial_state['question']}'")
    final_state = app.invoke(initial_state)

    print("\n==========================================")
    print("FINAL RESULT:")
    print(f"Generation:\n{final_state['generation']}")
    print(f"Total Loops Executed: {final_state['loop_count']}")
    print(f"Final Grounding Grade: {final_state['hallucination_grade']}")
    print("==========================================")

if __name__ == "__main__":
    test_multi_agent_rag()
