"""Iterative search loop - orchestrates search-reason-re-search for component selection."""
from amesim_builder.unified_search import get_searcher

MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.7

def run_iterative_selection(requirement, llm_review_fn=None):
    """Run iterative search-reason loop for a single component requirement.
    
    Args:
        requirement: dict with {functional_description, keywords, library, refined_constraints}
        llm_review_fn: optional function(keywords, candidates) -> {selected, confidence, suggested_keywords}
    
    Returns:
        dict with {selected, confidence, iterations, reasoning_log}
    """
    searcher = get_searcher()
    keywords = requirement.get("keywords", [requirement.get("functional_description", "")])
    if isinstance(keywords, list): keywords = keywords[0] if keywords else ""
    library = requirement.get("library")
    
    log = []
    for iteration in range(MAX_ITERATIONS):
        # Search
        candidates = searcher.search(keywords, top_k=5, library=library)
        if not candidates:
            log.append(f"Iter {iteration+1}: No results for '{keywords}'")
            if iteration == 0:
                keywords = requirement.get("functional_description", keywords)
                continue
            break
        
        # If we have behavior search, use it
        if hasattr(searcher, 'search_by_behavior'):
            candidates = searcher.search_by_behavior(keywords, top_k=5, library=library)
        
        # If no LLM review function, return top candidate
        if llm_review_fn is None:
            log.append(f"Iter {iteration+1}: Auto-selected top candidate (no LLM review)")
            return {
                "selected": candidates[0] if candidates else None,
                "confidence": candidates[0].get("rrf_score", 0.5) if candidates else 0,
                "iterations": iteration + 1,
                "reasoning_log": log
            }
        
        # LLM review
        review = llm_review_fn(keywords, candidates)
        log.append(f"Iter {iteration+1}: confidence={review.get('confidence', 0):.2f}")
        
        if review.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
            return {
                "selected": review.get("selected"),
                "confidence": review["confidence"],
                "iterations": iteration + 1,
                "reasoning_log": log
            }
        
        # Refine and re-search
        if review.get("suggested_keywords"):
            keywords = review["suggested_keywords"]
            log.append(f"  Refining search to: {keywords}")
    
    return {
        "selected": None,
        "confidence": 0,
        "iterations": MAX_ITERATIONS,
        "reasoning_log": log
    }
