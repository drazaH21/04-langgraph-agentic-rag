from typing import List, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Defines the shared state structure for our multi-agent self-correcting RAG graph.
    """
    question: str                 # Original user query
    rewritten_query: str          # Refined query for search optimization
    documents: List[str]          # Retrieved context chunks from hybrid search
    generation: str               # LLM generated response
    loop_count: int               # Self-correction loop counter
    hallucination_grade: Optional[str] # Grounding check grade ("yes" / "no")
