"""
Interactive KV Cache Statistics Dashboard
Real-time monitoring and profiling of cache performance

Features:
- Live cache statistics display
- Hit rate visualization
- Layer-by-layer analysis
- Memory usage tracking
- Performance bottleneck detection
- Comparative analysis (simple vs quantized)
"""

import torch
from quantized_kv_cache import QuantizedKVCache, NF4Quantizer
from simple_kv_cache import SimpleKVCache
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
import time
from collections import defaultdict


@dataclass
class LayerStats:
    """Statistics for a specific layer"""
    layer_id: int
    hits: int = 0
    misses: int = 0
    total_size_bytes: int = 0
    avg_dequant_time_ms: float = 0.0
    quantization_error: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0
    
    @property
    def size_mb(self) -> float:
        return self.total_size_bytes / 1024 / 1024


class CacheProfiler:
    """Profile cache performance in detail"""
    
    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        self.layer_stats: Dict[int, LayerStats] = {
            i: LayerStats(layer_id=i) for i in range(num_layers)
        }
        self.total_dequant_time = 0.0
        self.dequant_samples = 0
        self.profile_start = datetime.now()
    
    def record_layer_hit(self, layer_id: int, dequant_time_ms: float = 0.0):
        """Record a cache hit for a layer"""
        self.layer_stats[layer_id].hits += 1
        if dequant_time_ms > 0:
            self.total_dequant_time += dequant_time_ms
            self.dequant_samples += 1
            # Update average
            avg = self.total_dequant_time / self.dequant_samples
            self.layer_stats[layer_id].avg_dequant_time_ms = avg
    
    def record_layer_miss(self, layer_id: int):
        """Record a cache miss for a layer"""
        self.layer_stats[layer_id].misses += 1
    
    def record_layer_size(self, layer_id: int, size_bytes: int):
        """Record size of layer in cache"""
        self.layer_stats[layer_id].total_size_bytes = size_bytes
    
    def record_quantization_error(self, layer_id: int, error: float):
        """Record quantization error for a layer"""
        self.layer_stats[layer_id].quantization_error = error
    
    def get_profile_summary(self) -> Dict:
        """Get complete profile summary"""
        total_hits = sum(s.hits for s in self.layer_stats.values())
        total_misses = sum(s.misses for s in self.layer_stats.values())
        total_size = sum(s.total_size_bytes for s in self.layer_stats.values())
        avg_error = sum(s.quantization_error for s in self.layer_stats.values()) / len(self.layer_stats)
        
        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0,
            "total_cache_size_mb": total_size / 1024 / 1024,
            "avg_dequant_time_ms": self.total_dequant_time / self.dequant_samples if self.dequant_samples > 0 else 0,
            "avg_quantization_error": avg_error,
            "elapsed_seconds": (datetime.now() - self.profile_start).total_seconds(),
        }


class InteractiveDashboard:
    """Interactive dashboard for cache monitoring"""
    
    def __init__(self):
        self.profiler = CacheProfiler()
        self.layer_hit_history: Dict[int, List[float]] = defaultdict(list)
        self.memory_history: List[float] = []
        self.time_history: List[float] = []
    
    def print_layer_summary(self):
        """Print summary of each layer"""
        print("\n" + "="*80)
        print("LAYER-BY-LAYER ANALYSIS")
        print("="*80)
        print(f"{'Layer':<8} {'Hits':<10} {'Misses':<10} {'Hit Rate':<12} {'Size (MB)':<12} {'Avg Deq (ms)':<15} {'Quant Error':<12}")
        print("-"*80)
        
        for layer_id in range(self.profiler.num_layers):
            stats = self.profiler.layer_stats[layer_id]
            print(f"{layer_id:<8} {stats.hits:<10} {stats.misses:<10} {stats.hit_rate:>10.1f}% "
                  f"{stats.size_mb:>10.2f}    {stats.avg_dequant_time_ms:>13.3f}  {stats.quantization_error:>10.6f}")
    
    def print_performance_metrics(self):
        """Print performance metrics"""
        summary = self.profiler.get_profile_summary()
        
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        print(f"Total Cache Hits:         {summary['total_hits']:>10}")
        print(f"Total Cache Misses:       {summary['total_misses']:>10}")
        print(f"Overall Hit Rate:         {summary['overall_hit_rate']:>9.1f}%")
        print(f"Total Cache Size:         {summary['total_cache_size_mb']:>9.2f} MB")
        print(f"Avg Dequantization Time:  {summary['avg_dequant_time_ms']:>9.3f} ms")
        print(f"Avg Quantization Error:   {summary['avg_quantization_error']:>9.6f}")
        print(f"Elapsed Time:             {summary['elapsed_seconds']:>9.1f} s")
    
    def print_bottleneck_analysis(self):
        """Identify and print performance bottlenecks"""
        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS")
        print("="*80)
        
        summary = self.profiler.get_profile_summary()
        
        # Find layers with low hit rates
        low_hit_layers = []
        for layer_id in range(self.profiler.num_layers):
            stats = self.profiler.layer_stats[layer_id]
            if stats.hit_rate < 50 and (stats.hits + stats.misses) > 0:
                low_hit_layers.append((layer_id, stats.hit_rate))
        
        if low_hit_layers:
            print(f"\n⚠️  Low Hit Rate Layers:")
            for layer_id, hit_rate in sorted(low_hit_layers, key=lambda x: x[1]):
                print(f"   Layer {layer_id}: {hit_rate:.1f}% hit rate")
        else:
            print("\n✅ All layers have good hit rates (>50%)")
        
        # Find layers with high quantization error
        high_error_layers = []
        for layer_id in range(self.profiler.num_layers):
            stats = self.profiler.layer_stats[layer_id]
            if stats.quantization_error > 0.1:
                high_error_layers.append((layer_id, stats.quantization_error))
        
        if high_error_layers:
            print(f"\n⚠️  High Quantization Error Layers:")
            for layer_id, error in sorted(high_error_layers, key=lambda x: -x[1])[:5]:
                print(f"   Layer {layer_id}: Error = {error:.6f}")
        else:
            print("\n✅ All layers have acceptable quantization error (<0.1)")
        
        # Find layers with high dequantization overhead
        high_dequant_layers = []
        avg_dequant = summary['avg_dequant_time_ms']
        for layer_id in range(self.profiler.num_layers):
            stats = self.profiler.layer_stats[layer_id]
            if stats.avg_dequant_time_ms > avg_dequant * 1.5:
                high_dequant_layers.append((layer_id, stats.avg_dequant_time_ms))
        
        if high_dequant_layers:
            print(f"\n⚠️  High Dequantization Time Layers:")
            for layer_id, dequant_time in sorted(high_dequant_layers, key=lambda x: -x[1])[:5]:
                print(f"   Layer {layer_id}: {dequant_time:.3f} ms")
        else:
            print("\n✅ Dequantization time is consistent across layers")
    
    def print_recommendations(self):
        """Print optimization recommendations"""
        print("\n" + "="*80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        summary = self.profiler.get_profile_summary()
        
        # Check overall hit rate
        if summary['overall_hit_rate'] < 70:
            recommendations.append(
                "✓ Increase cache size (current hit rate < 70%)"
            )
        
        # Check dequantization time
        if summary['avg_dequant_time_ms'] > 1.0:
            recommendations.append(
                "✓ Consider GPU fused dequantization (current overhead > 1ms)"
            )
        
        # Check quantization error
        if summary['avg_quantization_error'] > 0.05:
            recommendations.append(
                "✓ Consider per-layer adaptive quantization (error > 0.05)"
            )
        
        # Check if cache is small
        if summary['total_cache_size_mb'] < 100:
            recommendations.append(
                "✓ Expand cache capacity for better statistics"
            )
        
        if not recommendations:
            recommendations.append("✅ Cache is well-optimized! No major improvements needed.")
        
        for rec in recommendations:
            print(rec)
    
    def print_comparison_table(self, simple_cache: SimpleKVCache, 
                              quantized_cache: QuantizedKVCache):
        """Print comparison between simple and quantized cache"""
        print("\n" + "="*80)
        print("SIMPLE vs QUANTIZED CACHE COMPARISON")
        print("="*80)
        
        simple_stats = simple_cache.get_stats()
        quantized_stats = quantized_cache.get_stats()
        
        print(f"{'Metric':<30} {'Simple Cache':<20} {'Quantized Cache':<20}")
        print("-"*80)
        print(f"{'Hit Rate':<30} {simple_stats['hit_rate_percent']:>18.1f}% {quantized_stats['hit_rate_percent']:>18.1f}%")
        print(f"{'Memory Used (MB)':<30} {simple_stats['size_mb'] or 0:>18.2f} {quantized_stats['size_mb'] or 0:>18.2f}")
        print(f"{'Time Saved (s)':<30} {simple_stats.get('total_time_saved_seconds', 0):>18.2f} {quantized_stats.get('total_time_saved_seconds', 0):>18.2f}")
        print(f"{'Entries Cached':<30} {simple_stats['cache_entries']:>18} {quantized_stats['cache_entries']:>18}")
        
        # Calculate memory efficiency
        if simple_stats['size_mb'] > 0 and quantized_stats['size_mb'] > 0:
            savings = (1 - quantized_stats['size_mb'] / simple_stats['size_mb']) * 100
            print(f"{'Memory Savings':<30} {'':<20} {savings:>18.1f}%")


def demo_interactive_dashboard():
    """Demonstrate the interactive dashboard"""
    print("\n" + "="*80)
    print("INTERACTIVE KV CACHE STATISTICS DASHBOARD")
    print("="*80)
    
    # Create caches
    simple_cache = SimpleKVCache(max_size_gb=5.0)
    quantized_cache = QuantizedKVCache(max_cache_size_mb=5120)
    dashboard = InteractiveDashboard()
    
    # Simulate activity
    print("\nSimulating cache activity...")
    
    batch_size = 2
    seq_len = 512
    num_layers = 32
    num_unique_prefixes = 10
    num_requests = 100
    
    quantizer = NF4Quantizer()
    
    for request_id in range(num_requests):
        prefix_idx = request_id % num_unique_prefixes
        prefix = torch.full((seq_len,), prefix_idx, dtype=torch.long)
        
        # Try quantized cache first
        cached = quantized_cache.get_all_layers(prefix, num_layers, "cpu")
        
        if cached is not None:
            # Cache hit - record for each layer
            for layer_id in range(num_layers):
                dashboard.profiler.record_layer_hit(layer_id, dequant_time_ms=0.05)
        else:
            # Cache miss - simulate forward pass
            kv_dict = {}
            for layer_id in range(num_layers):
                k = torch.randn(batch_size, 32, seq_len, 64)
                v = torch.randn(batch_size, 32, seq_len, 64)
                
                # Measure quantization error
                k_q, k_scale = quantizer.quantize_4bit(k.cpu())
                error = torch.mean((k - quantizer.dequantize_4bit(k_q, k_scale)) ** 2).item()
                
                dashboard.profiler.record_layer_miss(layer_id)
                dashboard.profiler.record_quantization_error(layer_id, error)
                
                kv_dict[layer_id] = (k, v)
            
            # Cache for next time
            quantized_cache.cache_all_layers(prefix, kv_dict, compute_time_ms_per_layer=10.0)
            
            # Record sizes
            for layer_id in range(num_layers):
                size = (kv_dict[layer_id][0].numel() + kv_dict[layer_id][1].numel()) * 4
                dashboard.profiler.record_layer_size(layer_id, size)
        
        if (request_id + 1) % 20 == 0:
            print(f"  Processed {request_id + 1}/{num_requests} requests")
    
    # Print results
    dashboard.print_layer_summary()
    dashboard.print_performance_metrics()
    dashboard.print_bottleneck_analysis()
    dashboard.print_recommendations()
    dashboard.print_comparison_table(simple_cache, quantized_cache)
    
    print("\n" + "="*80)
    print("Dashboard Demo Complete!")
    print("="*80 + "\n")


class RealTimeMonitor:
    """Real-time monitoring with live updates"""
    
    def __init__(self, update_interval: int = 10):
        self.update_interval = update_interval
        self.request_count = 0
        self.last_update = datetime.now()
    
    def should_update(self) -> bool:
        """Check if should update display"""
        if (datetime.now() - self.last_update).total_seconds() >= self.update_interval:
            self.last_update = datetime.now()
            return True
        return False
    
    def update_request_count(self):
        """Increment request count"""
        self.request_count += 1
    
    def print_live_status(self, cache: QuantizedKVCache, profiler: CacheProfiler):
        """Print live status update"""
        stats = cache.get_stats()
        summary = profiler.get_profile_summary()
        
        print(f"\r[Requests: {self.request_count:>4}] "
              f"[Hit Rate: {stats['hit_rate_percent']:>5.1f}%] "
              f"[Memory: {stats['size_mb']:>7.1f}MB] "
              f"[Dequant: {summary['avg_dequant_time_ms']:>6.3f}ms]", 
              end="", flush=True)


if __name__ == "__main__":
    demo_interactive_dashboard()
