"""Quick test to debug the issue."""

import torch
from simple_kv_cache import SimpleKVCache

cache = SimpleKVCache(max_size_gb=20)

# Cache one layer
k = torch.randn(1, 32, 2048, 64)
v = torch.randn(1, 32, 2048, 64)

prefix = "test prefix"
success = cache.cache_kv(prefix, layer=0, k_tensor=k, v_tensor=v)
print(f"Cached layer 0: {success}")

# Try to get all layers (should fail since only layer 0 is cached)
print("Trying get_all_layers with num_layers=2...")
result = cache.get_all_layers(prefix, num_layers=2)
print(f"Result (should be None): {result}")

print("Success!")
