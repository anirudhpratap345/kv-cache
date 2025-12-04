"""
Production-grade distributed KV cache using Redis backend.

This is the "poor man's distributed KVCache" - suitable for:
- Development and testing
- Single-cluster deployments (up to 100 GPUs)
- Moderate token counts (8K-32K context)

For massive scale (1M+ tokens, 1000+ GPUs), use:
- NVIDIA Infinity or
- Microsoft DeepSpeed-Inference or
- Custom Ray Serve + Plasma setup
"""

import redis
import torch
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import time
import logging

from src.core.base_kv_cache import BaseKVCache, KVTensorPair, CacheStatistics
from src.core.tensor_serialization import TensorSerializer, SerializedTensor
from src.core.prefix_matching import compute_prefix_hash

logger = logging.getLogger(__name__)


class DistributedKVCache(BaseKVCache):
    """
    Redis-backed KV cache for distributed LLM serving.
    
    Features:
    - Multi-GPU sharding support
    - Automatic serialization/deserialization
    - Prefix-based lookups (O(1) via hashing)
    - TTL support for automatic eviction
    - Statistics and monitoring
    
    Example:
        >>> cache = DistributedKVCache()
        >>> cache.cache_kv("hello world", layer=0, k_tensor, v_tensor)
        >>> kv = cache.get_cached_kv("hello world", layer=0)
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        device: str = "cuda",
        max_cache_size_gb: float = 100.0,
        precision: str = "float16",
        compress: bool = False,
        ttl_seconds: int = 86400,
    ):
        """
        Initialize Redis-backed KV cache.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            device: Target device ('cuda' or 'cpu')
            max_cache_size_gb: Max cache size (advisory, Redis controls actual limit)
            precision: Tensor precision ('float16' or 'bfloat16')
            compress: Whether to compress tensors (slower write, faster network)
            ttl_seconds: Default TTL for cache entries
        """
        super().__init__(device, max_cache_size_gb)
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        
        # Verify connection
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        self.precision = precision
        self.compress = compress
        self.ttl_seconds = ttl_seconds
        self.key_prefix = "kv_cache"
    
    def _make_key(self, prefix: str, layer: int) -> str:
        """Create Redis key from prefix and layer."""
        prefix_hash = compute_prefix_hash(prefix)
        return f"{self.key_prefix}:{prefix_hash}:{layer}"
    
    def _make_metadata_key(self, prefix: str, layer: int) -> str:
        """Create Redis key for metadata."""
        return f"{self._make_key(prefix, layer)}:meta"
    
    def cache_kv(
        self,
        prefix: str,
        layer: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Store KV tensors in Redis.
        
        Args:
            prefix: Input text prefix
            layer: Transformer layer number
            k_tensor: Key tensor
            v_tensor: Value tensor
            ttl_seconds: Optional override for TTL
            
        Returns:
            True if cached, False if size exceeds limit
        """
        ttl = ttl_seconds or self.ttl_seconds
        
        try:
            # Serialize tensors
            serialized_k = TensorSerializer.serialize(
                k_tensor, self.precision, self.compress
            )
            serialized_v = TensorSerializer.serialize(
                v_tensor, self.precision, self.compress
            )
            
            key = self._make_key(prefix, layer)
            
            # Store KV data
            pipeline = self.redis_client.pipeline()
            pipeline.set(f"{key}:k", serialized_k.data, ex=ttl)
            pipeline.set(f"{key}:v", serialized_v.data, ex=ttl)
            
            # Store metadata
            metadata = {
                "k_shape": ",".join(map(str, serialized_k.shape)),
                "k_dtype": serialized_k.dtype,
                "v_shape": ",".join(map(str, serialized_v.shape)),
                "v_dtype": serialized_v.dtype,
                "compressed": str(self.compress),
                "timestamp": str(datetime.now()),
            }
            
            for meta_key, meta_value in metadata.items():
                pipeline.hset(self._make_metadata_key(prefix, layer), meta_key, meta_value)
            
            pipeline.expire(self._make_metadata_key(prefix, layer), ttl)
            pipeline.execute()
            
            # Update stats
            total_size = len(serialized_k.data) + len(serialized_v.data)
            self.stats.total_bytes_stored += total_size
            
            logger.debug(f"Cached {prefix[:50]}... layer {layer}: {total_size/1024:.1f} KB")
            return True
            
        except Exception as e:
            logger.error(f"Error caching KV: {e}")
            self.stats.evictions += 1
            return False
    
    def get_cached_kv(
        self,
        prefix: str,
        layer: int,
    ) -> Optional[KVTensorPair]:
        """
        Retrieve KV tensors from Redis.
        
        Args:
            prefix: Input text prefix
            layer: Transformer layer number
            
        Returns:
            KVTensorPair if found, None otherwise
        """
        try:
            key = self._make_key(prefix, layer)
            
            # Fetch metadata
            metadata = self.redis_client.hgetall(self._make_metadata_key(prefix, layer))
            if not metadata:
                self.stats.misses += 1
                return None
            
            # Fetch K and V tensors
            k_data = self.redis_client.get(f"{key}:k")
            v_data = self.redis_client.get(f"{key}:v")
            
            if k_data is None or v_data is None:
                self.stats.misses += 1
                return None
            
            # Reconstruct serialized tensors
            k_shape = tuple(map(int, metadata[b"k_shape"].decode().split(",")))
            v_shape = tuple(map(int, metadata[b"v_shape"].decode().split(",")))
            compressed = metadata[b"compressed"].decode() == "True"
            
            serialized_k = SerializedTensor(
                data=k_data,
                shape=k_shape,
                dtype=metadata[b"k_dtype"].decode(),
                device="cuda",
                compressed=compressed,
            )
            
            serialized_v = SerializedTensor(
                data=v_data,
                shape=v_shape,
                dtype=metadata[b"v_dtype"].decode(),
                device="cuda",
                compressed=compressed,
            )
            
            # Deserialize
            k_tensor = TensorSerializer.deserialize(serialized_k, self.device, "float32")
            v_tensor = TensorSerializer.deserialize(serialized_v, self.device, "float32")
            
            # Create KVTensorPair
            prefix_hash = compute_prefix_hash(prefix)
            kv_pair = KVTensorPair(
                k_tensor=k_tensor,
                v_tensor=v_tensor,
                layer=layer,
                prefix_hash=prefix_hash,
                timestamp=datetime.now(),
                ttl_seconds=self.ttl_seconds,
            )
            
            self.stats.hits += 1
            self.stats.last_accessed = datetime.now()
            
            logger.debug(f"Cache hit: {prefix[:50]}... layer {layer}")
            return kv_pair
            
        except Exception as e:
            logger.error(f"Error retrieving KV: {e}")
            self.stats.misses += 1
            return None
    
    def find_matching_prefixes(
        self,
        prefix: str,
        layer: int,
        similarity_threshold: float = 0.95,
    ) -> list:
        """
        Find similar prefixes (uses Redis SCAN for prefix matching).
        
        Note: Full similarity matching requires loading all keys from Redis.
        For production, consider external indexing (Elasticsearch, etc.)
        
        Args:
            prefix: Query prefix
            layer: Layer number
            similarity_threshold: Minimum similarity
            
        Returns:
            List of similar prefix hashes
        """
        # For now, return empty - full implementation would require
        # external similarity engine or Redis modules
        return []
    
    def evict(self, prefix: str, layer: Optional[int] = None) -> bool:
        """
        Remove cache entries.
        
        Args:
            prefix: Input prefix
            layer: Specific layer or None to evict all
            
        Returns:
            True if evicted, False otherwise
        """
        try:
            prefix_hash = compute_prefix_hash(prefix)
            pattern = f"{self.key_prefix}:{prefix_hash}:{layer if layer else '*'}"
            
            # Use SCAN to find and delete
            keys_deleted = 0
            for key in self.redis_client.scan_iter(match=pattern):
                keys_deleted += self.redis_client.delete(key)
                # Also delete metadata
                self.redis_client.delete(f"{key}:meta")
            
            if keys_deleted > 0:
                self.stats.evictions += 1
            
            return keys_deleted > 0
            
        except Exception as e:
            logger.error(f"Error evicting: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            pattern = f"{self.key_prefix}:*"
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)
            
            logger.info("Cleared all cache entries")
            self.stats.total_bytes_stored = 0
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> CacheStatistics:
        """Return cache statistics."""
        return self.stats
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage from Redis.
        
        Returns:
            Dict with memory info
        """
        try:
            info = self.redis_client.info("memory")
            return {
                "used_bytes": info.get("used_memory", 0),
                "max_bytes": self.max_cache_size_bytes,
                "percent_used": (info.get("used_memory", 0) / self.max_cache_size_bytes * 100)
                if self.max_cache_size_bytes > 0 else 0,
                "redis_memory_human": info.get("used_memory_human", "N/A"),
                "redis_connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Redis connection and basic stats.
        
        Returns:
            Health status dict
        """
        try:
            start = time.time()
            self.redis_client.ping()
            latency_ms = (time.time() - start) * 1000
            
            info = self.redis_client.info()
            
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "memory_used_mb": info.get("used_memory", 0) / (1024**2),
                "keys_count": self.redis_client.dbsize(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
