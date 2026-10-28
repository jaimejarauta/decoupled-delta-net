"""KV-cache eviction vs. asymmetric quantization surgery for sparse-attention hybrid LLMs."""
from .compress import (
    ARMS_DEFAULT,
    attention_layer_indices,
    compress_kv_cache,
    forward_post_prefill,
    get_architecture_stats,
    measure_kv_and_state_bytes,
    prefill_with_salience,
)

__all__ = [
    "ARMS_DEFAULT",
    "attention_layer_indices",
    "compress_kv_cache",
    "forward_post_prefill",
    "get_architecture_stats",
    "measure_kv_and_state_bytes",
    "prefill_with_salience",
]

