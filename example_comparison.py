"""
Simple example: Compare inference WITH and WITHOUT KV cache.

Run this to see the speed difference!
"""

import torch
import time
import numpy as np
from typing import Dict, Any, Tuple
from simple_kv_cache import SimpleKVCache


def simulate_transformer_forward(
    batch_size: int = 1,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 64,
    compute_time_ms: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate transformer forward pass (just creates dummy tensors).
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        compute_time_ms: Simulated compute time
        
    Returns:
        (k_tensor, v_tensor)
    """
    # Simulate compute time
    time.sleep(compute_time_ms / 1000.0)
    
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    return k, v


def without_cache_simulation(num_requests: int = 100) -> Dict[str, Any]:
    """Simulate inference WITHOUT KV cache."""
    print(f"\n{'='*60}")
    print(f"SCENARIO 1: WITHOUT KV Cache (baseline)")
    print(f"{'='*60}")
    
    latencies = []
    total_time = 0
    
    # Repeated prompts (80% same prefix, 20% different)
    prefixes = [
        "Compare Next.js vs Remix for a marketing site",
        "What about Astro?",
        "Mobile considerations?",
    ]
    
    for i in range(num_requests):
        prefix = prefixes[i % len(prefixes)]
        
        request_start = time.time()
        
        # Always compute fresh (no cache)
        # Simulate: encoding + full generation
        k, v = simulate_transformer_forward(compute_time_ms=250)
        
        request_end = time.time()
        latency = (request_end - request_start) * 1000  # ms
        latencies.append(latency)
        total_time += latency
        
        if (i + 1) % 20 == 0:
            print(f"  Request {i+1}/{num_requests}: {latency:.1f}ms")
    
    latencies = np.array(latencies)
    
    return {
        'latencies': latencies,
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'avg': np.mean(latencies),
        'total_time': total_time,
        'throughput_tokens_per_sec': (num_requests * 100) / (total_time / 1000),  # Assuming 100 output tokens
    }


def with_cache_simulation(num_requests: int = 100) -> Dict[str, Any]:
    """Simulate inference WITH KV cache."""
    print(f"\n{'='*60}")
    print(f"SCENARIO 2: WITH KV Cache")
    print(f"{'='*60}")
    
    cache = SimpleKVCache(max_size_gb=10)
    
    latencies = []
    total_time = 0
    
    # Repeated prompts
    prefixes = [
        "Compare Next.js vs Remix for a marketing site",
        "What about Astro?",
        "Mobile considerations?",
    ]
    
    for i in range(num_requests):
        prefix = prefixes[i % len(prefixes)]
        
        request_start = time.time()
        
        # Try cache first
        kv = cache.get_cached_kv(prefix, layer=0)
        
        if kv is None:
            # Cache miss: compute and cache
            k, v = simulate_transformer_forward(compute_time_ms=250)
            cache.cache_kv(prefix, layer=0, k_tensor=k, v_tensor=v)
        else:
            # Cache hit: just decode output
            time.sleep(50 / 1000.0)  # 50ms to decode only
        
        request_end = time.time()
        latency = (request_end - request_start) * 1000  # ms
        latencies.append(latency)
        total_time += latency
        
        if (i + 1) % 20 == 0:
            stats = cache.get_stats()
            print(
                f"  Request {i+1}/{num_requests}: {latency:.1f}ms "
                f"(Hit rate: {stats['hit_rate_percent']:.0f}%)"
            )
    
    latencies = np.array(latencies)
    
    return {
        'latencies': latencies,
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'avg': np.mean(latencies),
        'total_time': total_time,
        'throughput_tokens_per_sec': (num_requests * 100) / (total_time / 1000),
        'cache_stats': cache.get_stats(),
    }


def main():
    """Run the comparison."""
    print("\n")
    print("=" * 60)
    print("Simple KV Cache Demonstration".center(60))
    print("Pure Python, No Redis".center(60))
    print("=" * 60)
    
    num_requests = 100
    
    # Baseline
    baseline = without_cache_simulation(num_requests)
    
    # With cache
    cached = with_cache_simulation(num_requests)
    
    print(f"\n{'='*60}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*60}\n")
    
    print(f"{'Metric':<30} {'Without Cache':<20} {'With Cache':<20}")
    print("-" * 70)
    
    print(
        f"{'P50 Latency (ms)':<30} "
        f"{baseline['p50']:<20.1f} "
        f"{cached['p50']:<20.1f}"
    )
    
    print(
        f"{'P95 Latency (ms)':<30} "
        f"{baseline['p95']:<20.1f} "
        f"{cached['p95']:<20.1f}"
    )
    
    print(
        f"{'P99 Latency (ms)':<30} "
        f"{baseline['p99']:<20.1f} "
        f"{cached['p99']:<20.1f}"
    )
    
    print(
        f"{'Average Latency (ms)':<30} "
        f"{baseline['avg']:<20.1f} "
        f"{cached['avg']:<20.1f}"
    )
    
    print(
        f"{'Throughput (tokens/s)':<30} "
        f"{baseline['throughput_tokens_per_sec']:<20.0f} "
        f"{cached['throughput_tokens_per_sec']:<20.0f}"
    )
    
    
    # Calculate speedups
    latency_speedup = baseline['avg'] / cached['avg']
    throughput_speedup = cached['throughput_tokens_per_sec'] / baseline['throughput_tokens_per_sec']
    
    print(f"\n{'-'*70}")
    
    print(f"\n[SPEEDUP] Latency improvement: {latency_speedup:.1f}x")
    print(f"[SPEEDUP] Throughput improvement: {throughput_speedup:.1f}x")
    
    if 'cache_stats' in cached:
        stats = cached['cache_stats']
        print(f"[CACHE] Hit rate: {stats['hit_rate_percent']:.1f}%")
        print(f"[CACHE] Size used: {stats['size_mb']:.1f}MB / {stats['max_size_mb']:.1f}MB")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
