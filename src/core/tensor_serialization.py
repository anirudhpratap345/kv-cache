"""
Efficient tensor serialization and deserialization for KV cache.

Key insight: Tensors must be:
1. Converted to low-precision (float16/bfloat16)
2. Serialized to bytes efficiently
3. Transmitted quickly
4. Deserialized to GPU memory
"""

import torch
import numpy as np
import io
import gzip
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SerializedTensor:
    """Container for serialized tensor data."""
    data: bytes
    shape: Tuple[int, ...]
    dtype: str
    device: str
    compressed: bool = False


class TensorSerializer:
    """Handle conversion between torch tensors and serialized bytes."""
    
    # Precision levels
    PRECISION_FLOAT32 = "float32"
    PRECISION_FLOAT16 = "float16"
    PRECISION_BFLOAT16 = "bfloat16"
    
    @staticmethod
    def to_low_precision(tensor: torch.Tensor, precision: str = "float16") -> torch.Tensor:
        """
        Convert tensor to lower precision for storage efficiency.
        
        Args:
            tensor: Input tensor (usually float32)
            precision: Target precision (float16, bfloat16)
            
        Returns:
            Tensor in lower precision format
        """
        if precision == "float16":
            return tensor.to(torch.float16)
        elif precision == "bfloat16":
            return tensor.to(torch.bfloat16)
        else:
            return tensor
    
    @staticmethod
    def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU for serialization."""
        return tensor.cpu().detach()
    
    @staticmethod
    def serialize(
        tensor: torch.Tensor,
        precision: str = "float16",
        compress: bool = False,
    ) -> SerializedTensor:
        """
        Serialize a tensor to bytes.
        
        Args:
            tensor: PyTorch tensor
            precision: Target precision
            compress: Whether to gzip compress
            
        Returns:
            SerializedTensor object
        """
        # Move to CPU and convert to lower precision
        tensor_cpu = TensorSerializer.to_cpu(tensor)
        tensor_fp = TensorSerializer.to_low_precision(tensor_cpu, precision)
        
        # Convert to numpy for serialization
        numpy_array = tensor_fp.numpy()
        
        # Serialize using torch.save for compatibility
        buffer = io.BytesIO()
        torch.save(numpy_array, buffer)
        data = buffer.getvalue()
        
        # Optional compression
        if compress:
            data = gzip.compress(data, compresslevel=3)  # Fast compression
        
        return SerializedTensor(
            data=data,
            shape=tuple(numpy_array.shape),
            dtype=str(numpy_array.dtype),
            device=tensor.device.type,
            compressed=compress,
        )
    
    @staticmethod
    def deserialize(
        serialized: SerializedTensor,
        target_device: str = "cuda",
        target_dtype: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Deserialize bytes back to tensor.
        
        Args:
            serialized: SerializedTensor object
            target_device: Target device ('cuda' or 'cpu')
            target_dtype: Optional target dtype (e.g., 'float32')
            
        Returns:
            PyTorch tensor on target device
        """
        # Decompress if needed
        data = serialized.data
        if serialized.compressed:
            data = gzip.decompress(data)
        
        # Load from bytes
        buffer = io.BytesIO(data)
        numpy_array = torch.load(buffer, weights_only=True)
        
        # Convert back to tensor
        tensor = torch.from_numpy(numpy_array)
        
        # Move to target device
        if target_device == "cuda":
            tensor = tensor.to("cuda")
        
        # Optionally convert back to original dtype
        if target_dtype == "float32":
            tensor = tensor.to(torch.float32)
        
        return tensor
    
    @staticmethod
    def estimate_size(tensor: torch.Tensor, compressed: bool = False) -> int:
        """
        Estimate serialized size in bytes.
        
        Args:
            tensor: PyTorch tensor
            compressed: Whether compression will be applied
            
        Returns:
            Estimated size in bytes
        """
        # Rough estimate: element_size * num_elements + torch overhead (~1KB)
        size = tensor.element_size() * tensor.nelement() + 1024
        
        if compressed:
            # Typical compression ratio for float16 tensors: ~70%
            size = int(size * 0.7)
        
        return size


class BatchTensorSerializer:
    """Handle serialization of multiple tensors (e.g., multi-layer KV cache)."""
    
    @staticmethod
    def serialize_batch(
        tensors: dict,  # {layer_id: (k_tensor, v_tensor)}
        precision: str = "float16",
        compress: bool = False,
    ) -> dict:
        """
        Serialize multiple tensors efficiently.
        
        Args:
            tensors: Dict mapping layer IDs to (k_tensor, v_tensor) tuples
            precision: Target precision
            compress: Whether to compress
            
        Returns:
            Dict of serialized tensors
        """
        serialized = {}
        for layer_id, (k_tensor, v_tensor) in tensors.items():
            serialized[f"{layer_id}_k"] = TensorSerializer.serialize(k_tensor, precision, compress)
            serialized[f"{layer_id}_v"] = TensorSerializer.serialize(v_tensor, precision, compress)
        
        return serialized
    
    @staticmethod
    def deserialize_batch(
        serialized: dict,
        target_device: str = "cuda",
    ) -> dict:
        """
        Deserialize multiple tensors.
        
        Args:
            serialized: Dict of SerializedTensor objects
            target_device: Target device
            
        Returns:
            Dict mapping layer IDs to (k_tensor, v_tensor) tuples
        """
        tensors = {}
        for key, serialized_tensor in serialized.items():
            tensor = TensorSerializer.deserialize(serialized_tensor, target_device)
            
            if key.endswith("_k"):
                layer_id = key[:-2]
                if layer_id not in tensors:
                    tensors[layer_id] = {}
                tensors[layer_id]["k"] = tensor
            elif key.endswith("_v"):
                layer_id = key[:-2]
                if layer_id not in tensors:
                    tensors[layer_id] = {}
                tensors[layer_id]["v"] = tensor
        
        # Convert to tuple format
        return {layer_id: (data["k"], data["v"]) for layer_id, data in tensors.items()}
    
    @staticmethod
    def compute_total_size(
        tensors: dict,
        precision: str = "float16",
        compress: bool = False,
    ) -> int:
        """
        Estimate total size of batch before serialization.
        
        Args:
            tensors: Dict of tensors
            precision: Target precision
            compress: Whether compression will be used
            
        Returns:
            Total estimated size in bytes
        """
        total = 0
        for layer_id, (k_tensor, v_tensor) in tensors.items():
            total += TensorSerializer.estimate_size(k_tensor, compress)
            total += TensorSerializer.estimate_size(v_tensor, compress)
        
        return total


# Benchmark utilities
def benchmark_serialization():
    """Quick benchmark of serialization performance."""
    print("Serialization Performance Benchmark\n" + "="*50)
    
    # Create dummy tensors (like real KV cache)
    batch_size, num_heads, seq_len, head_dim = 1, 32, 2048, 64
    k_tensor = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    v_tensor = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    
    import time
    
    # Benchmark float32
    start = time.time()
    serialized_fp32 = TensorSerializer.serialize(k_tensor, precision="float32", compress=False)
    elapsed_fp32 = time.time() - start
    
    # Benchmark float16
    start = time.time()
    serialized_fp16 = TensorSerializer.serialize(k_tensor, precision="float16", compress=False)
    elapsed_fp16 = time.time() - start
    
    # Benchmark with compression
    start = time.time()
    serialized_compressed = TensorSerializer.serialize(k_tensor, precision="float16", compress=True)
    elapsed_compressed = time.time() - start
    
    print(f"Original size (float32): {len(serialized_fp32.data) / 1024 / 1024:.2f} MB")
    print(f"  Serialization time: {elapsed_fp32*1000:.2f} ms")
    print(f"\nFloat16 size: {len(serialized_fp16.data) / 1024 / 1024:.2f} MB")
    print(f"  Serialization time: {elapsed_fp16*1000:.2f} ms")
    print(f"  Savings: {(1 - len(serialized_fp16.data)/len(serialized_fp32.data))*100:.1f}%")
    print(f"\nCompressed (gzip): {len(serialized_compressed.data) / 1024 / 1024:.2f} MB")
    print(f"  Serialization time: {elapsed_compressed*1000:.2f} ms")
    print(f"  Savings: {(1 - len(serialized_compressed.data)/len(serialized_fp32.data))*100:.1f}%")


if __name__ == "__main__":
    benchmark_serialization()
