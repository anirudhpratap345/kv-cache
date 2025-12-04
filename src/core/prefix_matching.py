"""
Prefix matching and hashing for KV cache lookups.

Key insight: Use SHA256 hashing for O(1) prefix lookups,
with optional fuzzy matching for approximate cache hits.
"""

import hashlib
from typing import List, Tuple
import torch
from difflib import SequenceMatcher


def compute_prefix_hash(prefix: str) -> str:
    """
    Compute SHA256 hash of prefix for fast lookups.
    
    Args:
        prefix: Input text prefix
        
    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(prefix.encode()).hexdigest()


def compute_token_hash(tokens: List[int]) -> str:
    """
    Compute hash from token IDs (useful if working at token level).
    
    Args:
        tokens: List of token IDs
        
    Returns:
        SHA256 hash
    """
    token_bytes = bytes(tokens)
    return hashlib.sha256(token_bytes).hexdigest()


def get_prefix_similarity(prefix1: str, prefix2: str) -> float:
    """
    Compute string similarity between two prefixes.
    Uses SequenceMatcher for approximate matching.
    
    Args:
        prefix1: First prefix
        prefix2: Second prefix
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Normalize by shorter prefix to avoid bias toward longer strings
    matcher = SequenceMatcher(None, prefix1, prefix2)
    return matcher.ratio()


def find_similar_prefixes(
    query_prefix: str,
    candidate_prefixes: List[str],
    threshold: float = 0.9,
    max_results: int = 5,
) -> List[Tuple[str, float]]:
    """
    Find prefixes similar to query prefix.
    
    Args:
        query_prefix: Query prefix to match
        candidate_prefixes: List of candidate prefixes
        threshold: Minimum similarity score
        max_results: Maximum number of results
        
    Returns:
        List of (prefix, similarity_score) tuples, sorted by similarity
    """
    results = []
    for candidate in candidate_prefixes:
        similarity = get_prefix_similarity(query_prefix, candidate)
        if similarity >= threshold:
            results.append((candidate, similarity))
    
    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_results]


def compute_prefix_length(prefix: str, vocab_size: int = 50000) -> int:
    """
    Estimate number of tokens in prefix.
    
    Rough heuristic: avg 4 chars per token (English)
    
    Args:
        prefix: Input prefix
        vocab_size: Vocabulary size (for reference)
        
    Returns:
        Estimated token count
    """
    return max(1, len(prefix) // 4)


def hierarchical_prefix_hash(prefix: str, num_levels: int = 3) -> List[str]:
    """
    Compute hierarchical prefix hashes for multi-level caching.
    
    Idea: Cache at multiple granularities
    - Level 0: Full prefix
    - Level 1: First 75% of prefix
    - Level 2: First 50% of prefix
    
    Args:
        prefix: Input prefix
        num_levels: Number of hierarchical levels
        
    Returns:
        List of hashes from full to increasingly shorter prefixes
    """
    hashes = []
    for level in range(num_levels):
        fraction = 1.0 - (level / num_levels)
        truncated_len = max(1, int(len(prefix) * fraction))
        truncated_prefix = prefix[:truncated_len]
        hashes.append(compute_prefix_hash(truncated_prefix))
    
    return hashes


class PrefixMatcher:
    """Utility class for prefix matching operations."""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.threshold = similarity_threshold
        self.prefix_cache: dict = {}  # prefix -> hash
    
    def add_prefix(self, prefix: str) -> str:
        """Add prefix and cache its hash."""
        prefix_hash = compute_prefix_hash(prefix)
        self.prefix_cache[prefix] = prefix_hash
        return prefix_hash
    
    def find_matches(
        self,
        query_prefix: str,
        include_exact: bool = True,
    ) -> List[Tuple[str, str, float]]:
        """
        Find matching prefixes in cache.
        
        Args:
            query_prefix: Query prefix
            include_exact: Whether to prioritize exact match
            
        Returns:
            List of (prefix, hash, similarity) tuples
        """
        results = []
        query_hash = compute_prefix_hash(query_prefix)
        
        for cached_prefix, cached_hash in self.prefix_cache.items():
            if include_exact and cached_hash == query_hash:
                # Exact match - highest priority
                results.insert(0, (cached_prefix, cached_hash, 1.0))
            else:
                similarity = get_prefix_similarity(query_prefix, cached_prefix)
                if similarity >= self.threshold:
                    results.append((cached_prefix, cached_hash, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)
        return results


# Benchmark utilities
def benchmark_prefix_hashing():
    """Benchmark prefix hashing performance."""
    print("Prefix Hashing Performance\n" + "="*50)
    
    import time
    
    # Generate test prefixes
    test_prefixes = [
        "Compare Next.js vs Remix for a marketing site",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming how we build software",
        "Distributed systems require careful handling of state",
    ] * 100  # 400 prefixes
    
    # Benchmark SHA256
    start = time.time()
    hashes = [compute_prefix_hash(p) for p in test_prefixes]
    elapsed = time.time() - start
    
    print(f"Hashed {len(test_prefixes)} prefixes in {elapsed*1000:.2f} ms")
    print(f"Average: {elapsed*1000000/len(test_prefixes):.2f} µs per prefix")
    
    # Benchmark similarity
    query = "Compare Next.js vs Remix for a marketing site"
    candidates = test_prefixes[:50]
    
    start = time.time()
    similarities = [get_prefix_similarity(query, c) for c in candidates]
    elapsed = time.time() - start
    
    print(f"\nComputed {len(candidates)} similarities in {elapsed*1000:.2f} ms")
    print(f"Average: {elapsed*1000000/len(candidates):.2f} µs per comparison")


if __name__ == "__main__":
    benchmark_prefix_hashing()
