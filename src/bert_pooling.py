def predict_with_chunks(..., top_k: int | None = 3, temperature: float = 1.0, eps: float = 1e-6):
    """
    Aggregates predictions from chunks.
    
    Supports the following methods:
    - mean
    - max
    - noisy_or
    """
    
    if aggregation_method == 'noisy_or':
        # Compute per-chunk probabilities via temperature-scaled softmax
        probs = softmax(chunk_logits / temperature)
        # Restrict to top_k highest p(label=1)
        if top_k is not None:
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]
        else:
            top_probs = probs
        # Apply clipping with eps
        clipped_probs = np.clip(top_probs, eps, 1.0)
        # Combine probabilities
        p_doc = 1 - np.prod(1 - clipped_probs)
        agg_logits = np.log([1 - p_doc, p_doc]).astype(np.float32)
        return agg_logits
    
    # existing mean/max behavior unchanged

    raise ValueError(f"Invalid aggregation method: {aggregation_method}. Use 'mean', 'max', or 'noisy_or'.")