"""
Simple, pure Python KV Cache - no Redis, no external dependencies beyond PyTorch.

Perfect for:
- Local development
- Single machine serving (< 100 concurrent users)
- Learning how KV caching works
- Prototyping before Redis deployment
"""

import torch
import hashlib
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single KV cache entry for one prefix+layer."""
    k_tensor: torch.Tensor
    v_tensor: torch.Tensor
    layer: int
    prefix: str
    timestamp: datetime
    ttl_seconds: int = 86400  # 24 hours default
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.now() > (self.timestamp + timedelta(seconds=self.ttl_seconds))
    
    def size_bytes(self) -> int:
        """Memory footprint in bytes."""
        return (
            self.k_tensor.element_size() * self.k_tensor.nelement() +
            self.v_tensor.element_size() * self.v_tensor.nelement()
        )


class SimpleKVCache:
    """
    Pure Python in-memory KV cache for LLM KV states.
    
    Simple, fast, no dependencies. Perfect for local development.
    
    Example:
        >>> cache = SimpleKVCache(max_size_gb=10)
        >>> cache.cache_kv("hello world", layer=0, k, v)
        >>> kv = cache.get_cached_kv("hello world", layer=0)
    """
    
    def __init__(self, max_size_gb: float = 10.0, device: str = "cpu"):
        """
        Initialize KV cache.
        
        Args:
            max_size_gb: Maximum cache size in GB
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self._cache: Dict[str, CacheEntry] = {}
        self._current_size = 0
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_time_saved': 0.0,
        }
    
    def _make_key(self, prefix: str, layer: int) -> str:
        """Create cache key from prefix and layer."""
        prefix_hash = hashlib.sha256(prefix.encode()).hexdigest()
        return f"{prefix_hash}:{layer}"
    
    def cache_kv(
        self,
        prefix: str,
        layer: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        ttl_seconds: int = 86400,
    ) -> bool:
        """
        Cache KV tensors.
        
        Args:
            prefix: Input text prefix
            layer: Transformer layer number
            k_tensor: Key tensor
            v_tensor: Value tensor
            ttl_seconds: Time to live
            
        Returns:
            True if cached, False if size exceeded
        """
        # Move to CPU for caching (smaller memory footprint)
        k = k_tensor.cpu().detach()
        v = v_tensor.cpu().detach()
        
        entry = CacheEntry(
            k_tensor=k,
            v_tensor=v,
            layer=layer,
            prefix=prefix,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds,
        )
        
        entry_size = entry.size_bytes()
        
        # Check if we have space
        if self._current_size + entry_size > self.max_size_bytes:
            logger.warning(
                f"Cache full: {self._current_size/1024/1024:.1f}MB + "
                f"{entry_size/1024/1024:.1f}MB > {self.max_size_bytes/1024/1024:.1f}MB"
            )
            self.stats['evictions'] += 1
            return False
        
        # Store
        key = self._make_key(prefix, layer)
        self._cache[key] = entry
        self._current_size += entry_size
        
        logger.debug(f"Cached {prefix[:50]}... layer {layer}: {entry_size/1024/1024:.1f}MB")
        return True
    
    def get_cached_kv(
        self,
        prefix: str,
        layer: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve cached KV tensors.
        
        Args:
            prefix: Input text prefix
            layer: Transformer layer number
            
        Returns:
            (k_tensor, v_tensor) if found and valid, None otherwise
        """
        key = self._make_key(prefix, layer)
        
        if key not in self._cache:
            self.stats['misses'] += 1
            return None
        
        entry = self._cache[key]
        
        # Check expiry
        if entry.is_expired():
            del self._cache[key]
            self._current_size -= entry.size_bytes()
            self.stats['evictions'] += 1
            self.stats['misses'] += 1
            return None
        
        # Hit!
        self.stats['hits'] += 1
        
        # Move to target device
        k = entry.k_tensor.to(self.device)
        v = entry.v_tensor.to(self.device)
        
        logger.debug(f"Cache hit: {prefix[:50]}... layer {layer}")
        return (k, v)
    
    def get_all_layers(
        self,
        prefix: str,
        num_layers: int = 32,
    ) -> Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Retrieve all layers for a prefix (or None if any missing).
        
        Args:
            prefix: Input prefix
            num_layers: Number of layers to retrieve
            
        Returns:
            Dict mapping layer_id to (k, v) tuples, or None if incomplete
        """
        result = {}
        
        for layer in range(num_layers):
            kv = self.get_cached_kv(prefix, layer)
            if kv is None:
                return None  # Incomplete cache
            result[layer] = kv
        
        return result
    
    def cache_all_layers(
        self,
        prefix: str,
        kv_tensors: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        """
        Cache all layers at once.
        
        Args:
            prefix: Input prefix
            kv_tensors: Dict mapping layer_id to (k, v) tuples
            
        Returns:
            True if all cached successfully
        """
        all_cached = True
        
        for layer, (k, v) in kv_tensors.items():
            success = self.cache_kv(prefix, layer, k, v)
            if not success:
                all_cached = False
        
        return all_cached
    
    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._current_size = 0
        logger.info("Cache cleared")
    
    def evict(self, prefix: str, layer: Optional[int] = None) -> bool:
        """Remove specific entry or all layers for a prefix."""
        prefix_hash = hashlib.sha256(prefix.encode()).hexdigest()
        
        if layer is not None:
            key = f"{prefix_hash}:{layer}"
            if key in self._cache:
                size = self._cache[key].size_bytes()
                del self._cache[key]
                self._current_size -= size
                return True
        else:
            # Remove all layers for this prefix
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix_hash)]
            for key in keys_to_remove:
                size = self._cache[key].size_bytes()
                del self._cache[key]
                self._current_size -= size
            return len(keys_to_remove) > 0
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'hit_rate_percent': hit_rate,
            'total_requests': total_requests,
            'size_mb': self._current_size / 1024 / 1024,
            'max_size_mb': self.max_size_bytes / 1024 / 1024,
            'size_percent': (self._current_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0,
            'time_saved_seconds': self.stats['total_time_saved'],
        }
    
    def print_stats(self) -> None:
        """Print formatted statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("KV CACHE STATISTICS")
        print("="*60)
        print(f"Hit Rate: {stats['hit_rate_percent']:.1f}% ({stats['hits']} hits, {stats['misses']} misses)")
        print(f"Evictions: {stats['evictions']}")
        print(f"Size: {stats['size_mb']:.1f}MB / {stats['max_size_mb']:.1f}MB ({stats['size_percent']:.1f}%)")
        print(f"Time Saved: {stats['time_saved_seconds']:.2f}s")
        print("="*60 + "\n")
