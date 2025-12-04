"""
Abstract base classes and interfaces for KV cache implementations.

This module defines the contract that all KV cache backends must implement,
ensuring interoperability and consistent behavior across different strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
from datetime import datetime, timedelta


@dataclass
class CacheStatistics:
    """Track cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_bytes_stored: int = 0
    last_accessed: Optional[datetime] = None
    
    @property
    def hit_rate(self) -> float:
        """Compute hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def __str__(self) -> str:
        return (
            f"Cache Stats:\n"
            f"  Hit Rate: {self.hit_rate:.2f}%\n"
            f"  Hits: {self.hits}, Misses: {self.misses}, Evictions: {self.evictions}\n"
            f"  Total Size: {self.total_bytes_stored / (1024**3):.2f} GB"
        )


@dataclass
class KVTensorPair:
    """Represents a KV cache entry."""
    k_tensor: torch.Tensor
    v_tensor: torch.Tensor
    layer: int
    prefix_hash: str
    timestamp: datetime
    ttl_seconds: int = 86400  # Default 24 hours
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > (self.timestamp + timedelta(seconds=self.ttl_seconds))
    
    def size_bytes(self) -> int:
        """Compute memory footprint of this KV pair."""
        return self.k_tensor.element_size() * self.k_tensor.nelement() + \
               self.v_tensor.element_size() * self.v_tensor.nelement()


class BaseKVCache(ABC):
    """Abstract base class for all KV cache implementations."""
    
    def __init__(self, device: str = "cuda", max_cache_size_gb: float = 100.0):
        """
        Initialize base KV cache.
        
        Args:
            device: 'cuda' or 'cpu'
            max_cache_size_gb: Maximum cache size in GB
        """
        self.device = device
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024**3)
        self.stats = CacheStatistics()
    
    @abstractmethod
    def cache_kv(
        self,
        prefix: str,
        layer: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        ttl_seconds: int = 86400,
    ) -> bool:
        """
        Store KV tensors in cache.
        
        Args:
            prefix: Input text prefix
            layer: Transformer layer number
            k_tensor: Key tensor [batch, heads, seq_len, head_dim]
            v_tensor: Value tensor [batch, heads, seq_len, head_dim]
            ttl_seconds: Time to live for this cache entry
            
        Returns:
            True if cached successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_cached_kv(
        self,
        prefix: str,
        layer: int,
    ) -> Optional[KVTensorPair]:
        """
        Retrieve cached KV tensors.
        
        Args:
            prefix: Input text prefix
            layer: Transformer layer number
            
        Returns:
            KVTensorPair if found and valid, None otherwise
        """
        pass
    
    @abstractmethod
    def find_matching_prefixes(
        self,
        prefix: str,
        layer: int,
        similarity_threshold: float = 0.95,
    ) -> list:
        """
        Find similar prefixes in cache for approximate matching.
        
        Args:
            prefix: Input text prefix
            layer: Transformer layer number
            similarity_threshold: Minimum similarity (0.0-1.0)
            
        Returns:
            List of matching prefix hashes and their similarity scores
        """
        pass
    
    @abstractmethod
    def evict(self, prefix: str, layer: Optional[int] = None) -> bool:
        """
        Remove cache entry.
        
        Args:
            prefix: Input text prefix
            layer: Specific layer or None to evict all layers
            
        Returns:
            True if evicted, False if not found
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStatistics:
        """Return current cache statistics."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Return memory usage breakdown.
        
        Returns:
            Dict with 'used_bytes', 'max_bytes', 'percent_used'
        """
        pass


class LocalKVCache(BaseKVCache):
    """
    Simple in-memory KV cache for prototyping and testing.
    Not recommended for production (no persistence, single-machine only).
    """
    
    def __init__(self, device: str = "cuda", max_cache_size_gb: float = 100.0):
        super().__init__(device, max_cache_size_gb)
        self._cache: Dict[str, KVTensorPair] = {}
    
    def cache_kv(
        self,
        prefix: str,
        layer: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        ttl_seconds: int = 86400,
    ) -> bool:
        from src.core.prefix_matching import compute_prefix_hash
        
        prefix_hash = compute_prefix_hash(prefix)
        key = f"{prefix_hash}:{layer}"
        
        # Check size
        kv_pair = KVTensorPair(
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            layer=layer,
            prefix_hash=prefix_hash,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds,
        )
        
        if self.stats.total_bytes_stored + kv_pair.size_bytes() > self.max_cache_size_bytes:
            self.stats.evictions += 1
            return False
        
        self._cache[key] = kv_pair
        self.stats.total_bytes_stored += kv_pair.size_bytes()
        return True
    
    def get_cached_kv(
        self,
        prefix: str,
        layer: int,
    ) -> Optional[KVTensorPair]:
        from src.core.prefix_matching import compute_prefix_hash
        
        prefix_hash = compute_prefix_hash(prefix)
        key = f"{prefix_hash}:{layer}"
        
        if key in self._cache:
            kv_pair = self._cache[key]
            if not kv_pair.is_expired():
                self.stats.hits += 1
                self.stats.last_accessed = datetime.now()
                return kv_pair
            else:
                # Remove expired entry
                del self._cache[key]
                self.stats.evictions += 1
        
        self.stats.misses += 1
        return None
    
    def find_matching_prefixes(
        self,
        prefix: str,
        layer: int,
        similarity_threshold: float = 0.95,
    ) -> list:
        # Simplified: exact match only for in-memory cache
        return []
    
    def evict(self, prefix: str, layer: Optional[int] = None) -> bool:
        from src.core.prefix_matching import compute_prefix_hash
        
        prefix_hash = compute_prefix_hash(prefix)
        
        if layer is not None:
            key = f"{prefix_hash}:{layer}"
            if key in self._cache:
                self.stats.total_bytes_stored -= self._cache[key].size_bytes()
                del self._cache[key]
                return True
        else:
            # Evict all layers for this prefix
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix_hash)]
            for key in keys_to_remove:
                self.stats.total_bytes_stored -= self._cache[key].size_bytes()
                del self._cache[key]
            return len(keys_to_remove) > 0
        
        return False
    
    def clear(self) -> None:
        self._cache.clear()
        self.stats.total_bytes_stored = 0
    
    def get_stats(self) -> CacheStatistics:
        return self.stats
    
    def get_memory_usage(self) -> Dict[str, Any]:
        return {
            "used_bytes": self.stats.total_bytes_stored,
            "max_bytes": self.max_cache_size_bytes,
            "percent_used": (self.stats.total_bytes_stored / self.max_cache_size_bytes * 100),
        }
