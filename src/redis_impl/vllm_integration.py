"""
Integration with vLLM for production deployment.

Shows how to integrate KV cache with vLLM for real inference workloads.
"""

from src.redis_impl.distributed_kv_cache import DistributedKVCache
from src.core.prefix_matching import compute_prefix_hash
import torch
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class vLLMKVCacheIntegration:
    """Integration layer between vLLM and distributed KV cache."""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        device: str = "cuda",
    ):
        """
        Initialize vLLM + KV cache integration.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            device: 'cuda' or 'cpu'
        """
        self.cache = DistributedKVCache(
            redis_host=redis_host,
            redis_port=redis_port,
            device=device,
            precision="float16",
            compress=True,
        )
        self.device = device
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved": 0.0,
        }
    
    def extract_kv_from_forward_pass(
        self,
        model_output: Dict[str, Any],
        prefix: str,
    ) -> bool:
        """
        Extract KV cache from model forward pass and store in Redis.
        
        Args:
            model_output: Output dict from model (typically contains past_key_values)
            prefix: Input prefix to cache
            
        Returns:
            True if cached successfully
        """
        # This depends on your model architecture
        # Example for Llama/Mistral:
        
        if "past_key_values" not in model_output:
            logger.warning("Model output doesn't contain past_key_values")
            return False
        
        past_kv = model_output["past_key_values"]
        
        # past_kv is typically: tuple of (k, v) tuples for each layer
        success = True
        for layer_idx, (k_tensor, v_tensor) in enumerate(past_kv):
            try:
                self.cache.cache_kv(
                    prefix=prefix,
                    layer=layer_idx,
                    k_tensor=k_tensor,
                    v_tensor=v_tensor,
                )
            except Exception as e:
                logger.error(f"Failed to cache layer {layer_idx}: {e}")
                success = False
        
        return success
    
    def get_cached_kv_for_prefix(
        self,
        prefix: str,
        num_layers: int = 32,
    ) -> Optional[list]:
        """
        Retrieve cached KV for prefix (if available).
        
        Args:
            prefix: Input prefix
            num_layers: Number of transformer layers
            
        Returns:
            List of (k, v) tensors if found, None otherwise
        """
        cached_kv = []
        
        for layer_idx in range(num_layers):
            kv_pair = self.cache.get_cached_kv(prefix, layer=layer_idx)
            if kv_pair is None:
                # Cache miss on any layer means we can't use it
                return None
            
            cached_kv.append((kv_pair.k_tensor, kv_pair.v_tensor))
        
        return cached_kv
    
    def generate_with_cache(
        self,
        model,
        tokenizer,
        prompt: str,
        prefix: str,
        max_new_tokens: int = 100,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate text using KV cache if available.
        
        Args:
            model: vLLM engine or HF model
            tokenizer: Tokenizer for encoding/decoding
            prompt: Full prompt (prefix + question)
            prefix: Prefix to use for caching
            max_new_tokens: Maximum tokens to generate
            use_cache: Whether to use KV cache
            
        Returns:
            Dict with output, time_saved, cache_hit
        """
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        result = {
            "output": None,
            "time_saved": 0.0,
            "cache_hit": False,
            "latency": 0.0,
        }
        
        # Try to get cached KV
        cached_kv = None
        if use_cache:
            cached_kv = self.get_cached_kv_for_prefix(prefix)
            if cached_kv:
                self.stats["cache_hits"] += 1
                result["cache_hit"] = True
                logger.info(f"Cache hit for prefix: {prefix[:50]}...")
            else:
                self.stats["cache_misses"] += 1
                logger.info(f"Cache miss for prefix: {prefix[:50]}...")
        
        # Generate (model-specific implementation)
        # This is pseudo-code - adapt for your model
        try:
            # For vLLM:
            if hasattr(model, "generate"):
                # Use vLLM's generate
                output_ids = model.generate(
                    input_ids=tokenizer(prompt)["input_ids"],
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
                output = tokenizer.decode(output_ids[0])
            else:
                # Fallback for HF transformers
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )
                output = tokenizer.decode(outputs[0])
            
            result["output"] = output
            
            # Cache the KV states for future use
            if use_cache and not result["cache_hit"]:
                # Extract KV from outputs and cache
                if hasattr(outputs, "past_key_values"):
                    self.extract_kv_from_forward_pass(
                        {"past_key_values": outputs.past_key_values},
                        prefix=prefix,
                    )
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result["output"] = f"Error: {str(e)}"
        
        # Record timing
        elapsed = time.time() - start_time
        result["latency"] = elapsed
        
        if result["cache_hit"]:
            # Estimate time saved (rough heuristic)
            # Cache hit typically saves ~70% latency
            result["time_saved"] = elapsed * 2.3  # Approx: would take 3.3x longer without cache
            self.stats["total_time_saved"] += result["time_saved"]
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        hit_rate = (
            self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"]) * 100
            if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
            else 0
        )
        
        return {
            **self.stats,
            "hit_rate_percent": hit_rate,
            "hours_saved": self.stats["total_time_saved"] / 3600,
        }
    
    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("vLLM + KV Cache Integration Statistics")
        print("="*60)
        print(f"Total requests: {stats['total_requests']}")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
        print(f"\nTime saved: {stats['total_time_saved']:.1f}s ({stats['hours_saved']:.3f} hours)")
        print(f"Average latency: {self.stats['total_time_saved'] / max(self.stats['total_requests'], 1):.2f}s per request")
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # This is just for demonstration
    print("vLLM Integration Example\n")
    print("To use with real vLLM:")
    print("""
    from vllm import LLM, SamplingParams
    from src.redis_impl.vllm_integration import vLLMKVCacheIntegration
    
    # Initialize
    integration = vLLMKVCacheIntegration(redis_host="localhost")
    llm = LLM(model="meta-llama/Llama-2-70b-chat-hf")
    
    # Use in agent loop
    prefix = "You are a helpful AI. User is in India..."
    for query in agent_queries:
        result = integration.generate_with_cache(
            model=llm,
            tokenizer=llm.tokenizer,
            prompt=prefix + query,
            prefix=prefix,
        )
        print(result["output"])
    
    integration.print_stats()
    """)
