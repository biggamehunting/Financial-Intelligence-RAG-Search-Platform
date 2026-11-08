from threading import Lock

from FlagEmbedding import FlagReranker


MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_reranker = None
_reranker_lock = Lock()


def _get_reranker() -> FlagReranker:
    """Load the reranker once and reuse it for subsequent requests."""
    global _reranker

    if _reranker is None:
        with _reranker_lock:
            # Check again after acquiring the lock so concurrent first
            # requests do not load multiple copies of the model.
            if _reranker is None:
                print(f"Loading local reranker: {MODEL_NAME}")
                _reranker = FlagReranker(
                    MODEL_NAME,
                    use_fp16=False,  # CPU-compatible; use_fp16=True with supported GPU
                )
                print("Local BGE reranker loaded.")

    return _reranker


def rerank_documents(question: str, documents, top_n: int = 3):
    """
    Rerank LangChain Documents locally with a cross-encoder.

    Returns the selected original Document objects and their reranker scores,
    preserving metadata and document identity for downstream citations.
    """
    if not documents or top_n <= 0:
        return [], []

    reranker = _get_reranker()
    pairs = [
        [question, document.page_content]
        for document in documents
    ]

    print("Reranking documents with local BGE cross-encoder...")
    scores = reranker.compute_score(pairs, normalize=True)

    # FlagEmbedding normally returns a list for a list of pairs. Normalize
    # the single-result case as well to keep the return type consistent.
    if isinstance(scores, (int, float)):
        scores = [scores]

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    reranked_documents = [document for document, _ in ranked_results]
    rerank_scores = [float(score) for _, score in ranked_results]

    for rank, (document, score) in enumerate(ranked_results, start=1):
        print(f"RERANK RESULT -> rank={rank}, score={score:.4f}")

    print("Local BGE reranking completed.")
    return reranked_documents, rerank_scores
