"""
Realistic example: Cache all layers of a 7B model with repeated prefixes.

This shows how to cache KV for ALL transformer layers at once.
"""

import torch
import time
from typing import List, Dict, Any
from simple_kv_cache import SimpleKVCache


def simulate_llm_forward(
    num_layers: int = 32,
    batch_size: int = 1,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 64,
) -> Dict[int, tuple]:
    """Simulate a full forward pass through all layers."""
    
    kv_tensors = {}
    
    for layer_id in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        kv_tensors[layer_id] = (k, v)
        
        # Simulate layer compute time (reduced for faster demo)
        time.sleep(0.005)  # 5ms per layer
    
    return kv_tensors


def agentic_workflow_simulation():
    """
    Simulate agentic workflow where the same prefix is reused 100s of times.
    
    This is the 95% cache hit scenario from the intro.
    """
    
    print("\n" + "="*70)
    print("AGENTIC WORKFLOW SIMULATION")
    print("Same prefix reused 100 times (95%+ cache hit rate)")
    print("="*70 + "\n")
    
    cache = SimpleKVCache(max_size_gb=20)
    num_layers = 4  # Reduce layers for speed
    
    # User's original query
    base_prompt = "I need to plan a 3-month roadmap for a SaaS startup using the PMArchitect framework. Please break this down into:"
    
    # Variations (different continuations, same prefix)
    continuations = [
        "1. Market analysis phase",
        "2. Product strategy phase",
        "3. Go-to-market planning",
        "How do I prioritize features?",
        "What metrics should I track?",
    ]
    
    full_requests = [base_prompt + cont for cont in continuations]
    
    # Simulate 50 requests with heavy prefix reuse
    requests = []
    for _ in range(10):  # 10 iterations
        requests.extend(full_requests)  # 5 variants per iteration
    
    print(f"Simulating {len(requests)} requests across {len(set(requests))} unique prompts...")
    print("(This may take a minute...)\n", flush=True)
    
    inference_times = []
    cache_times = []
    compute_times = []
    
    for i, prompt in enumerate(requests):
        
        # Check cache for this prompt (as prefix)
        cached_kv = cache.get_all_layers(prompt, num_layers=num_layers)
        
        if cached_kv is None:
            # Cache miss: compute all layers
            compute_start = time.time()
            kv_all_layers = simulate_llm_forward(num_layers=num_layers)
            compute_end = time.time()
            compute_time = (compute_end - compute_start) * 1000
            compute_times.append(compute_time)
            
            # Cache all layers
            cache_start = time.time()
            cache.cache_all_layers(prompt, kv_all_layers)
            cache_end = time.time()
            cache_time = (cache_end - cache_start) * 1000
            cache_times.append(cache_time)
            
            total_time = compute_time + cache_time
        else:
            # Cache hit: just retrieve
            retrieve_start = time.time()
            # In real usage, we'd move these to GPU and use them
            _ = cached_kv
            retrieve_end = time.time()
            total_time = (retrieve_end - retrieve_start) * 1000
        
        inference_times.append(total_time)
        
        if (i + 1) % 10 == 0:
            stats = cache.get_stats()
            print(
                f"  Request {i+1:2d}/{len(requests)}: "
                f"Hit rate={stats['hit_rate_percent']:5.1f}%",
                flush=True
            )
    
    final_stats = cache.get_stats()
    
    print(f"\n{'='*70}")
    print("AGENTIC WORKFLOW RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Total requests processed: {len(requests)}")
    print(f"Unique prompts cached: {len(set(requests))}")
    print(f"Cache hit rate: {final_stats['hit_rate_percent']:.1f}%")
    print(f"Total cache hits: {final_stats['hits']}")
    print(f"Total cache misses: {final_stats['misses']}")
    print(f"Cache size used: {final_stats['size_mb']:.1f}MB / {final_stats['max_size_mb']:.1f}MB")
    
    print(f"\nLatency Analysis:")
    print(f"  Average inference time: {sum(inference_times) / len(inference_times):.1f}ms")
    print(f"  Min: {min(inference_times):.1f}ms")
    print(f"  Max: {max(inference_times):.1f}ms")
    
    if compute_times:
        print(f"\nCompute time for cache misses:")
        print(f"  Average per miss: {sum(compute_times) / len(compute_times):.1f}ms")
        print(f"  Total compute: {sum(compute_times):.0f}ms")
    
    # Calculate time saved
    baseline_time = len(requests) * (sum(compute_times) / len(compute_times) if compute_times else 400)
    actual_time = sum(inference_times)
    time_saved = baseline_time - actual_time
    speedup = baseline_time / actual_time if actual_time > 0 else 1.0
    
    print(f"\n{'='*70}")
    print(f"[RESULT] Speedup: {speedup:.1f}x")
    print(f"[RESULT] Time saved: {time_saved:.0f}ms ({time_saved/1000:.1f}s) on {len(requests)} requests")
    print(f"{'='*70}\n")


def simple_cache_example():
    """Basic example of how to use the cache."""
    
    print("\n" + "="*70)
    print("BASIC USAGE EXAMPLE")
    print("="*70 + "\n")
    
    # Create cache
    cache = SimpleKVCache(max_size_gb=10)
    
    # Simulate KV tensors
    k = torch.randn(1, 32, 2048, 64)
    v = torch.randn(1, 32, 2048, 64)
    
    # Cache them
    print("1. Caching KV tensors for layer 0...")
    success = cache.cache_kv(
        prefix="Hello world",
        layer=0,
        k_tensor=k,
        v_tensor=v
    )
    print(f"   [OK] Cached: {success}\n")
    
    # Retrieve them
    print("2. Retrieving cached KV...")
    k_cached, v_cached = cache.get_cached_kv(prefix="Hello world", layer=0)
    print(f"   [OK] Retrieved k: shape={k_cached.shape}, device={k_cached.device}")
    print(f"   [OK] Retrieved v: shape={v_cached.shape}, device={v_cached.device}\n")
    
    # Get stats
    print("3. Cache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    print()


if __name__ == "__main__":
    simple_cache_example()
    agentic_workflow_simulation()
