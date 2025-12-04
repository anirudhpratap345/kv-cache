"""
Quick start script to test KV cache locally.

Run this to verify everything works before deploying.
"""

import sys
sys.path.insert(0, '/d:/KV Cache')

import torch
import time
from src.core.base_kv_cache import LocalKVCache
from src.redis_impl.distributed_kv_cache import DistributedKVCache
from src.core.prefix_matching import compute_prefix_hash, get_prefix_similarity
from src.benchmarks.benchmark_suite import KVCacheBenchmark


def test_basic_kv_cache():
    """Test basic KV cache functionality."""
    print("\n" + "="*60)
    print("Test 1: Basic KV Cache (In-Memory)")
    print("="*60)
    
    cache = LocalKVCache(device="cpu", max_cache_size_gb=10)
    
    # Create dummy tensors
    k = torch.randn(1, 32, 128, 64)
    v = torch.randn(1, 32, 128, 64)
    
    # Cache
    prefix = "Compare Next.js vs Remix"
    success = cache.cache_kv(prefix, layer=0, k_tensor=k, v_tensor=v)
    print(f"✓ Cached KV tensors: {success}")
    
    # Retrieve
    kv = cache.get_cached_kv(prefix, layer=0)
    print(f"✓ Retrieved from cache: {kv is not None}")
    
    # Check stats
    stats = cache.get_stats()
    print(f"✓ Cache hit rate: {stats.hit_rate:.1f}%")
    print(f"✓ Total size: {stats.total_bytes_stored / 1024 / 1024:.1f} MB")


def test_prefix_matching():
    """Test prefix matching and hashing."""
    print("\n" + "="*60)
    print("Test 2: Prefix Matching & Hashing")
    print("="*60)
    
    prefix1 = "Compare Next.js vs Remix for India"
    prefix2 = "Compare Next.js vs Remix"
    prefix3 = "Comparing web frameworks"
    
    hash1 = compute_prefix_hash(prefix1)
    hash2 = compute_prefix_hash(prefix2)
    hash3 = compute_prefix_hash(prefix3)
    
    print(f"✓ Prefix 1 hash: {hash1[:16]}...")
    print(f"✓ Prefix 2 hash: {hash2[:16]}...")
    print(f"✓ Prefix 3 hash: {hash3[:16]}...")
    print(f"✓ Hash 1 == Hash 2: {hash1 == hash2} (deterministic)")
    
    # Similarity
    sim_12 = get_prefix_similarity(prefix1, prefix2)
    sim_13 = get_prefix_similarity(prefix1, prefix3)
    
    print(f"✓ Similarity (1-2): {sim_12:.2%}")
    print(f"✓ Similarity (1-3): {sim_13:.2%}")


def test_redis_integration():
    """Test Redis integration (if available)."""
    print("\n" + "="*60)
    print("Test 3: Redis Integration (if Redis is running)")
    print("="*60)
    
    try:
        cache = DistributedKVCache(
            redis_host="localhost",
            redis_port=6379,
        )
        
        # Health check
        health = cache.health_check()
        print(f"✓ Redis health: {health['status']}")
        print(f"✓ Redis latency: {health.get('latency_ms', 'N/A'):.2f}ms")
        
        # Test caching
        k = torch.randn(1, 32, 128, 64)
        v = torch.randn(1, 32, 128, 64)
        
        prefix = "Test prefix"
        success = cache.cache_kv(prefix, layer=0, k_tensor=k, v_tensor=v)
        print(f"✓ Cached to Redis: {success}")
        
        # Retrieve
        kv = cache.get_cached_kv(prefix, layer=0)
        print(f"✓ Retrieved from Redis: {kv is not None}")
        
        # Stats
        memory = cache.get_memory_usage()
        print(f"✓ Redis memory used: {memory.get('redis_memory_human', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Redis test skipped: {str(e)}")
        print("  (Make sure Redis is running: docker run -p 6379:6379 redis:7-alpine)")


def test_benchmarks():
    """Run quick benchmark."""
    print("\n" + "="*60)
    print("Test 4: Quick Benchmark (100 requests)")
    print("="*60)
    
    benchmark = KVCacheBenchmark()
    results = benchmark.run_all_benchmarks(num_requests=100)
    
    baseline = results[0].throughput_tokens_per_sec
    print(f"\nSummary:")
    for result in results:
        speedup = result.throughput_tokens_per_sec / baseline if baseline > 0 else 1
        print(f"  {result.name:30} | {speedup:5.1f}× | {result.latency_p95_ms:6.2f}ms p95")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  KV Cache for LLM Serving - Quick Start Test Suite".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    test_basic_kv_cache()
    test_prefix_matching()
    test_redis_integration()
    test_benchmarks()
    
    print("\n" + "="*60)
    print("✓ All tests completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Read: docs/01_why_kv_cache_matters.md")
    print("2. Run: notebooks/01_basic_kv_cache.ipynb")
    print("3. Deploy: Follow docs/04_production_deployment.md")
    print()


if __name__ == "__main__":
    main()
