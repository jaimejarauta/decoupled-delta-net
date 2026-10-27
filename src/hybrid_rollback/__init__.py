"""Synchronized online checkpoint rollback and subspace overwrite probes for hybrid LLMs."""
from .cache_state import (
    RollbackMetrics,
    capture_recurrent_snapshot,
    count_hybrid_cache_layers,
    crop_kv_in_place,
    prefill_tokens,
    restore_recurrent_in_place,
    rollback_attention_only_kv,
    rollback_synchronized,
)
from .live_decode import LiveDecodeResult, live_decode_with_rollback
from .revolve import OnlineCheckpointPool, RecurrentSnapshot
from .subspace_probe import measure_recurrent_state_drift, measure_subspace_overlap

__all__ = [
    "LiveDecodeResult",
    "OnlineCheckpointPool",
    "RecurrentSnapshot",
    "RollbackMetrics",
    "capture_recurrent_snapshot",
    "count_hybrid_cache_layers",
    "crop_kv_in_place",
    "live_decode_with_rollback",
    "measure_recurrent_state_drift",
    "measure_subspace_overlap",
    "prefill_tokens",
    "restore_recurrent_in_place",
    "rollback_attention_only_kv",
    "rollback_synchronized",
]
