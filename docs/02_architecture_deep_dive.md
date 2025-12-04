# Architecture Deep Dive: How Distributed KV Cache Works

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Request                             │
│  "Compare Next.js vs Remix for a marketing site in India"      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  1. Compute Prefix Hash       │
         │  sha256(prefix) =             │
         │  "a3f72c1e..."                │
         └────────────┬──────────────────┘
                      │
         ┌────────────▼──────────────────┐
         │  2. Lookup in KV Store        │
         └────┬──────────────────┬───────┘
              │                  │
         ┌────▼────┐      ┌──────▼────┐
         │ HIT (80%)│      │ MISS (20%)│
         └────┬────┘      └──────┬────┘
              │                  │
       ┌──────▼──────────┐     ┌─▼──────────────────┐
       │ Load from Cache │     │ Generate Missing  │
       │ GPU/Redis/Disk  │     │ KV States         │
       │ (100-1000ms)    │     │ (2000-5000ms)     │
       └──────┬──────────┘     └─┬──────────────────┘
              │                  │
              └──────────┬───────┘
                         │
                    ┌────▼──────────────┐
                    │ 3. Decode Output  │
                    │ Tokens            │
                    │ (500ms)           │
                    └────┬───────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Result:      │
                  │ 0.8s vs 4.2s │
                  └──────────────┘
```

## Layer 1: In-Process GPU Cache (Local)

### Memory Layout

```
GPU Memory (80 GB on A100)
┌─────────────────────────────────────────────┐
│ Free Space                                  │ 10 GB
├─────────────────────────────────────────────┤
│ Model Weights (70B model)                  │ 140 GB (off-GPU on server)
├─────────────────────────────────────────────┤
│ KV Cache (Paged)                           │ 30 GB
│ ├─ Page 0: Tokens 0-2K for request 1      │
│ ├─ Page 1: Tokens 2K-4K for request 1     │
│ ├─ Page 2: Tokens 0-2K for request 2      │
│ └─ Page 3: Tokens 2K-4K for request 2     │
├─────────────────────────────────────────────┤
│ Batch Activations                           │ 10 GB
└─────────────────────────────────────────────┘
```

### Paged Attention Algorithm

Unlike vLLM, traditional transformers store KV cache contiguously:

```
Traditional (Wasteful)
┌─────────────────────────────────────┐
│ KV for request 1 (allocated: 100GB) │ Only using 50GB
│ KV for request 2 (allocated: 100GB) │ Only using 30GB
└─────────────────────────────────────┘
Total waste: 120GB

PagedAttention (Efficient)
┌──────┬──────┬──────┬──────┬──────┐
│Page 0│Page 1│Page 2│Page 3│Page 4│ ← Same 100GB
│Req 1 │Req 1 │Req 2 │Req 2 │Empty │
├──────┼──────┼──────┼──────┼──────┤
└──────────────────────────────────┘
Total waste: 20GB (only 1 empty page)
```

**Implementation in PyTorch:**

```python
# PagedAttention simulation
class PagedKVCache:
    def __init__(self, page_size=2048, num_pages=1000):
        self.page_size = page_size
        self.pages = {}  # page_id -> tensor
        self.free_pages = set(range(num_pages))
        
    def allocate_pages(self, seq_len):
        """Allocate pages for new sequence."""
        num_pages_needed = (seq_len + self.page_size - 1) // self.page_size
        page_ids = []
        
        for _ in range(num_pages_needed):
            page_id = self.free_pages.pop()
            page_ids.append(page_id)
            self.pages[page_id] = torch.zeros(
                1, 32, self.page_size, 64,  # [batch, heads, seq, head_dim]
                device="cuda", dtype=torch.float16
            )
        
        return page_ids
    
    def get_kv_tensors(self, page_ids):
        """Concatenate paged tensors back to contiguous."""
        k_tensors = [self.pages[pid][:, :, :, :64] for pid in page_ids]
        v_tensors = [self.pages[pid][:, :, :, 64:] for pid in page_ids]
        
        return torch.cat(k_tensors, dim=2), torch.cat(v_tensors, dim=2)
```

## Layer 2: Redis Hot Cache

### Serialization Strategy

**Challenge:** GPU tensors are huge (100-150 GB for single request) and in exotic formats.

**Solution:** Serialize smartly

```
GPU Tensor (float32)         Redis Storage
┌─────────────────┐          ┌──────────────┐
│ [1,32,8K,64]    │  ─────→  │ float16      │ ÷2 space
│ 16 GB           │          │ 8 GB         │
└─────────────────┘          └──────────────┘
    
With gzip compression
┌──────────────┐
│ compressed   │ ÷3 space
│ 2.7 GB       │ (typical ratio)
└──────────────┘
```

**Serialization Pipeline:**

```python
import torch
import io
import gzip

# 1. Lower precision
k_tensor = k_tensor.to(torch.float16)  # float32 → float16

# 2. Move to CPU
k_tensor_cpu = k_tensor.cpu()

# 3. Serialize
buffer = io.BytesIO()
torch.save(k_tensor_cpu.numpy(), buffer)
serialized = buffer.getvalue()  # ~8 GB

# 4. Compress
compressed = gzip.compress(serialized, compresslevel=3)  # ~2.7 GB

# 5. Store in Redis
redis_client.set(f"kv:{prefix_hash}:0:k", compressed, ex=86400)
# ex=86400 → 24 hour TTL
```

### Redis Key Structure

```
kv:a3f72c1e...:0:k          # Layer 0 key tensor
kv:a3f72c1e...:0:v          # Layer 0 value tensor
kv:a3f72c1e...:0:meta       # Metadata (shape, dtype, etc)
kv:a3f72c1e...:1:k          # Layer 1 key tensor
...
kv:a3f72c1e...:31:v         # Layer 31 value tensor
```

### Single Request KV Size

```
Model: 70B parameters, 32 layers
Context: 32K tokens
Precision: float16

Per layer: 1 × 32 heads × 32K tokens × 64 head_dim × 2 bytes
         = 128 MB per layer

Total: 32 layers × 128 MB = 4.1 GB
With compression: ~1.4 GB
```

**vs full 100B context:**
```
Context: 100K tokens
Per layer: 1 × 32 × 100K × 64 × 2 = 400 MB
Total: 32 × 400 = 12.8 GB
Compressed: ~4.3 GB
```

## Layer 3: Distributed Multi-GPU Cache

### Sharding Strategy

For massive contexts (>100K tokens) across 8 GPUs:

```
Horizontal Sharding (by sequence position)
┌─────────────────────────────────────────────────┐
│ GPU 0: All layers, seq 0-12.5K tokens          │
│ GPU 1: All layers, seq 12.5K-25K tokens        │
│ GPU 2: All layers, seq 25K-37.5K tokens        │
│ GPU 3: All layers, seq 37.5K-50K tokens        │
│ ...
│ GPU 7: All layers, seq 87.5K-100K tokens       │
└─────────────────────────────────────────────────┘
```

**Pros:** 
- Simple to implement
- Each GPU responsible for fixed slice
- Easy load balancing

**Cons:**
- Attention still needs cross-GPU communication
- Network bandwidth becomes bottleneck

### Vertical Sharding (by layer)

```
Vertical Sharding (by layer)
┌─────────────────────────────┐
│ GPU 0: Layers 0-3           │
│ GPU 1: Layers 4-7           │
│ GPU 2: Layers 8-11          │
│ ...
│ GPU 7: Layers 28-31         │
└─────────────────────────────┘
```

**Pros:**
- Can handle very large contexts
- Parallelizes layer computation naturally

**Cons:**
- Must gather activations between GPUs
- More complex synchronization

### Actual Implementation (NVIDIA Infinity)

NVIDIA Infinity uses **hybrid sharding**:

```python
# Pseudo-code from NVIDIA Infinity
class InfinityCache:
    def __init__(self, num_gpus=8):
        self.gpus = num_gpus
        self.layer_to_gpu = {}  # Map layers to GPUs
        self.seq_ranges = {}    # Map seq ranges to GPUs
    
    def store_kv(self, layer, seq_range, k_tensor, v_tensor):
        """Store KV with smart placement."""
        # Layer sharding: which GPU stores this layer?
        gpu_for_layer = layer % self.gpus
        
        # Seq sharding: which GPU stores this seq range?
        gpu_for_seq = (seq_range[0] // 12500) % self.gpus
        
        # Place on GPU that has fewer recent updates
        target_gpu = self._select_optimal_gpu(gpu_for_layer, gpu_for_seq)
        
        # Store compressed
        self.gpus[target_gpu].store(
            k_tensor.to(torch.float16).cpu(),
            compress=True
        )
    
    def retrieve_kv(self, layer, seq_range):
        """Retrieve from potentially multiple GPUs."""
        gpu_shards = []
        for gpu_id in self._find_replica_gpus(layer, seq_range):
            shard = self.gpus[gpu_id].fetch()
            gpu_shards.append(shard)
        
        # Reconstruct full tensor
        return torch.cat(gpu_shards, dim=2)
```

## Prefix Matching in Detail

### Hash-Based Lookup (O(1))

```python
import hashlib

prefix1 = "Compare Next.js vs Remix for a marketing site in India"
prefix2 = "Compare Next.js vs Remix for a marketing site"  # Slightly shorter

hash1 = hashlib.sha256(prefix1.encode()).hexdigest()
hash2 = hashlib.sha256(prefix2.encode()).hexdigest()

# These are COMPLETELY different!
# hash1 = "a3f72c1e..."
# hash2 = "b9d45e72..."

# So exact prefix matching won't work for similar queries
```

### Approximate Matching (O(n) but necessary)

```python
from difflib import SequenceMatcher

def similarity(s1, s2):
    """Compute string similarity."""
    return SequenceMatcher(None, s1, s2).ratio()

# Find similar prefixes in cache
query = "Compare Next.js vs Remix"
candidates = [
    "Compare Next.js vs Remix for a marketing site",  # 0.96 similar
    "Compare Remix vs Next.js",                       # 0.73 similar
    "Comparing web frameworks",                       # 0.45 similar
]

for cand in candidates:
    sim = similarity(query, cand)
    if sim > 0.9:  # threshold
        print(f"Use partial cache from: {cand}")
```

**Trade-off:** 
- Exact matching: Fast (O(1)) but fewer hits
- Approximate matching: Slower (O(n)) but more hits

**Best practice:** Use exact for first pass, approximate as fallback

## Fault Tolerance

### Replication

```
Primary: Redis Node 1
├─ kv:a3f72c1e...:0:k [8 GB]
├─ kv:a3f72c1e...:0:v [8 GB]
└─ ...

Replica: Redis Node 2 (auto-sync)
├─ kv:a3f72c1e...:0:k [8 GB]  ← Same data
├─ kv:a3f72c1e...:0:v [8 GB]
└─ ...

If Node 1 fails:
- Node 2 becomes primary
- Clients fail-over with < 100ms pause
- No cache data loss
```

### Graceful Degradation

```
Request arrives
    │
    ├─ Try cache lookup
    │  └─ If hit: use (0.1s)
    │  └─ If miss or Redis down: fall back to full generation (3.5s)
    │
    └─ Always return correct answer
```

## Performance Metrics

### Latency Breakdown (with full cache)

```
Request: "What about [new question]?"

Timeline:
  0ms   - Request arrives
 10ms   - Hash prefix
 50ms   - Network: fetch from Redis
150ms   - Decompress KV tensors
250ms   - Deserialize to GPU tensors
300ms   - Load to GPU memory
350ms   - Generate 100 output tokens
800ms   - Return to user

Total: 800ms vs 4200ms (baseline)
```

### Memory Overhead

```
Real scenario: 8 GPU cluster, 70B model

GPU Memory per GPU:
  - Model weights: 140 GB (off-GPU on server)
  - Batch activations: 10 GB
  - KV cache (paged): 20 GB
  - Free: 10 GB
  = 80 GB A100 GPU ✓

Redis Memory (for hot prefixes):
  - 1000 hot prefixes × 1.4 GB = 1.4 TB
  - Replication factor 2 = 2.8 TB
  - Cost: ~$250/hour on AWS

Total cost reduction: 70-80% → ROI in days
```

## Next: Production Deployment

Continue to: `04_production_deployment.md`

This will cover:
1. Monitoring & observability
2. Cache invalidation strategies
3. Multi-region replication
4. Cost optimization
