from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents import (
    retrieve_node,
    generate_node,
    hallucination_grader_node,
    query_rewriter_node
)

def decide_to_generate(state: AgentState) -> str:
    """
    Determines whether to re-write the query or proceed based on loop count.
    """
    print("[Router] Evaluating next step...")
    if state.get("loop_count", 0) > 2:
        print("[Router] Max loops reached. Forcing generation.")
        return "generate"
    return "generate"

def grade_generation_router(state: AgentState) -> str:
    """
    Conditional edge: if answer is grounded, finish. If not, re-try retrieval or rewrite.
    """
    grade = state.get("hallucination_grade")
    loops = state.get("loop_count", 0)
    
    if grade == "yes":
        print("[Router] Decision: Answer is grounded. Finishing workflow.")
        return "finish"
    elif loops <= 2:
        print("[Router] Decision: Answer ungrounded. Routing to query rewriter for self-correction loop.")
        return "rewrite"
    else:
        print("[Router] Decision: Max retries exceeded. Finishing workflow.")
        return "finish"

# Build the LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_hallucination", hallucination_grader_node)
workflow.add_node("rewrite_query", query_rewriter_node)

# Set entry point
workflow.set_entry_point("retrieve")

# Add edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade_hallucination")

# Add conditional edge for self-correction loop
workflow.add_conditional_edges(
    "grade_hallucination",
    grade_generation_router,
    {
        "finish": END,
        "rewrite": "rewrite_query"
    }
)

workflow.add_edge("rewrite_query", "retrieve")

# Compile graph
app = workflow.compile()
print("[LangGraph] Multi-agent self-correcting RAG graph compiled successfully.")
