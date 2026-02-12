import numpy as np
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def find_best_match(
    query_embedding: list,
    users: list,
    threshold: float = 0.85,
) -> dict:
    """
    Find the best matching user for the given embedding.

    Args:
        query_embedding: Query embedding vector
        users: List of User model instances
        threshold: Minimum similarity for acceptance

    Returns:
        dict with match result
    """
    best_score = -1.0
    best_user = None

    for user in users:
        stored_embedding = user.get_embedding()
        score = cosine_similarity(query_embedding, stored_embedding)

        logger.debug(f"User {user.roll_number}: similarity = {score:.4f}")

        if score > best_score:
            best_score = score
            best_user = user

    matched = best_score >= threshold

    result = {
        "matched": matched,
        "similarity_score": round(best_score, 4),
        "threshold": threshold,
        "status": "Authenticated" if matched else "Rejected",
    }

    if matched and best_user:
        result["user"] = best_user.to_dict()

    logger.info(
        f"Match result: {result['status']} "
        f"(score={best_score:.4f}, threshold={threshold})"
    )

    return result
