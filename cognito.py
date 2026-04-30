"""
cognito.py — Application-level KV cache paging for long-context LLM inference.

Works with any HuggingFace AutoModelForCausalLM + AutoTokenizer on CUDA,
MPS (Apple Silicon), or CPU, without custom CUDA kernels or runtime forks.

Three eviction policies (all plug-and-play):
  • RAGAwarePager   — segment-aware by RRF score + Ebbinghaus decay [recommended]
  • H2OEvictionPolicy — H2O heavy-hitter oracle via L2-norm proxy (Zhang 2023)
  • VirtualPageManager — StreamingLLM-style sink/recent window (Xiao 2024)

Quick start:
    from cognito import CognitoEngine, RAGAwarePager, ContextSegment

    pager = RAGAwarePager(threshold_gb=5.5)
    engine = CognitoEngine(model, tokenizer, pager=pager)

    result = engine.chat(
        messages=[{"role": "user", "content": "Who wrote Hamlet?"}],
        segments=[
            ContextSegment("Shakespeare wrote Hamlet around 1600.", score=0.95),
            ContextSegment("Hamlet is set in Denmark.", score=0.3),
        ],
        max_new_tokens=100,
    )
    print(result.text)

References:
  Kwon et al. 2023 — PagedAttention (SOSP)
  Agrawal et al. 2024 — Sarathi-Serve / chunked prefill (OSDI)
  Zhang et al. 2023 — H2O Heavy-Hitter Oracle (NeurIPS)
  Xiao et al. 2024 — StreamingLLM / attention sinks (ICLR)
  Devoto et al. 2024 — L2-norm KV eviction proxy (EMNLP)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "VirtualPageManager",
    "AdaptiveVirtualPageManager",
    "PredictiveMemoryPolicy",
    "RAGAwarePager",
    "H2OEvictionPolicy",
    "ContextSegment",
    "GenerationResult",
    "CognitoEngine",
    "load_model",
]

import gc
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

# ── Lazy torch proxy ──────────────────────────────────────────────────────────
# `import cognito` does NOT import torch.
# The proxy below replaces itself in the module globals on the very first
# attribute access (e.g. torch.zeros, torch.device), so the rest of the code
# uses `torch.xxx` exactly as written — no changes needed elsewhere.
# From the second access onward the proxy is gone and lookups are O(1).

class _TorchProxy:
    """Replaces itself with the real torch module on first attribute access."""
    __slots__ = ()

    def __getattr__(self, name: str):
        try:
            import torch as _real
        except ImportError as exc:
            raise ImportError(
                "cognito requires PyTorch >= 2.0.\n"
                "Install:  pip install torch\n"
                "CUDA:     pip install torch "
                "--index-url https://download.pytorch.org/whl/cu121"
            ) from exc
        # Replace this proxy in the module's global namespace so all
        # subsequent `torch.xxx` calls bypass the proxy entirely.
        globals()["torch"] = _real
        return getattr(_real, name)


if TYPE_CHECKING:
    import torch  # type: ignore[import-untyped]  # checkers see real module; not executed at runtime
else:
    torch = _TorchProxy()  # noqa: F811 — runtime lazy proxy, replaces itself on first use


def _check_versions() -> None:
    """
    Soft version check: warns if installed libraries are older than recommended.
    Never raises — calling code surfaces real errors if features are missing.
    Called automatically by CognitoEngine.__init__.
    """
    t = sys.modules.get("torch")
    if t is not None:
        ver = tuple(int(x) for x in t.__version__.split(".")[:2] if x.isdigit())
        if ver < (2, 0):
            warnings.warn(f"cognito: torch {t.__version__} detected; >= 2.0 recommended.")

    tf = sys.modules.get("transformers")
    if tf is not None:
        ver = tuple(int(x) for x in tf.__version__.split(".")[:2] if x.isdigit())
        if ver < (4, 46):
            warnings.warn(
                f"cognito: transformers {tf.__version__} detected. "
                "DynamicCache requires >= 4.46; Gemma 4 requires >= 5.5."
            )


# ─── Device / memory utilities ───────────────────────────────────────────────

def _infer_device(model=None) -> torch.device:
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _memory_allocated_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 ** 3
    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024 ** 3
        except Exception:
            return 0.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 1024 ** 3
    except ImportError:
        return 0.0


def _memory_peak_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 3
    if device.type == "mps":
        try:
            return torch.mps.driver_allocated_memory() / 1024 ** 3
        except Exception:
            return 0.0
    return 0.0


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _to_cpu(tensor: torch.Tensor, pinned: bool = True) -> torch.Tensor:
    """Copy tensor to CPU. Uses pinned memory on CUDA for async transfers."""
    use_pin = pinned and torch.cuda.is_available()
    buf = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu",
                      pin_memory=use_pin)
    buf.copy_(tensor.detach(), non_blocking=use_pin)
    return buf


# ─── KV cache utilities ───────────────────────────────────────────────────────

def _get_kv_tensors(cache) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extract (key_list, value_list) from any DynamicCache variant.
    Supports transformers >=4.36 legacy tuple format and >=4.46 DynamicCache.
    """
    # transformers >=4.46 DynamicCache with .key_cache list
    if hasattr(cache, "key_cache") and isinstance(getattr(cache, "key_cache"), list):
        kc = cache.key_cache
        if kc and hasattr(kc[0], "shape"):
            return kc, cache.value_cache

    # Some 5.x variants store layers as objects
    if hasattr(cache, "layers") and cache.layers:
        layer0 = cache.layers[0]
        if hasattr(layer0, "keys") and layer0.keys is not None:
            keys = [l.keys for l in cache.layers if l.keys is not None]
            vals = [l.values for l in cache.layers if l.values is not None]
            if keys:
                return keys, vals

    # Legacy: tuple of (k, v) per layer
    try:
        k_list = [layer[0] for layer in cache]
        v_list = [layer[1] for layer in cache]
        return k_list, v_list
    except (TypeError, IndexError):
        pass

    raise ValueError(f"Cannot extract KV tensors from cache type: {type(cache)}")


def _rebuild_cache(key_list: List[torch.Tensor], value_list: List[torch.Tensor]):
    """Reconstruct a DynamicCache from per-layer (k, v) lists."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(zip(key_list, value_list)):
        cache.update(k, v, i)
    return cache


def get_cache_seq_len(cache) -> int:
    """Return the sequence length currently stored in the KV cache."""
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    try:
        keys, _ = _get_kv_tensors(cache)
        if keys:
            return keys[0].shape[-2]
    except Exception:
        pass
    return 0


# ─── Eviction policies ────────────────────────────────────────────────────────

class VirtualPageManager:
    """
    StreamingLLM-style eviction (Xiao et al., ICLR 2024).

    Keeps `sink_size` attention-sink tokens + a recent window of size
    (page_size - sink_size). Middle tokens are offloaded to pinned CPU memory
    when VRAM exceeds `threshold_gb`.

    This is the base class. Use RAGAwarePager for RAG use cases or
    H2OEvictionPolicy for attention-heuristic eviction.
    """

    def __init__(self, threshold_gb: float = 7.5, page_size: int = 1024,
                 sink_size: int = 4):
        self.threshold_bytes = threshold_gb * 1024 ** 3
        self.page_size = page_size
        self.sink_size = sink_size
        self.cpu_swap: list = []
        self.active = False
        self._offloads = 0
        self._logged = False

    # ── Public helpers ────────────────────────────────────────────────────

    def reset(self) -> None:
        self.cpu_swap.clear()
        self.active = False
        self._offloads = 0
        self._logged = False

    @property
    def streamllm_offloads(self) -> int:
        return self._offloads

    @property
    def blocks_in_cpu(self) -> int:
        return len(self.cpu_swap)

    @property
    def eviction_count(self) -> int:
        return self._offloads

    # ── Internal ──────────────────────────────────────────────────────────

    def _pressure(self, device: torch.device) -> bool:
        return _memory_allocated_gb(device) * 1024 ** 3 > self.threshold_bytes

    def offload_kv_cache(self, cache, device: Optional[torch.device] = None):
        """Trim the middle of the KV cache to fit within threshold."""
        if not self.active or cache is None:
            return cache
        dev = device or _infer_device()
        if not self._pressure(dev):
            return cache

        keys, vals = _get_kv_tensors(cache)
        if not any(k.shape[-2] > self.page_size for k in keys):
            return cache

        new_k, new_v = [], []
        n_paged = 0
        keep = self.page_size - self.sink_size
        for i, (k, v) in enumerate(zip(keys, vals)):
            seq = k.shape[-2]
            if seq > self.page_size:
                self.cpu_swap.append((i, _to_cpu(k[:, :, self.sink_size:-keep, :]),
                                         _to_cpu(v[:, :, self.sink_size:-keep, :])))
                new_k.append(torch.cat([k[:, :, :self.sink_size, :],
                                        k[:, :, -keep:, :]], dim=2))
                new_v.append(torch.cat([v[:, :, :self.sink_size, :],
                                        v[:, :, -keep:, :]], dim=2))
                n_paged += 1
            else:
                new_k.append(k)
                new_v.append(v)

        _sync(dev)
        _empty_cache(dev)
        new_cache = _rebuild_cache(new_k, new_v)
        if not self._logged and n_paged:
            print(f"[Cognito/StreamingLLM] offload layers={n_paged} "
                  f"seq={get_cache_seq_len(new_cache)}")
            self._logged = True
        if n_paged:
            self._offloads += 1
        return new_cache


class AdaptiveVirtualPageManager(VirtualPageManager):
    """
    VirtualPageManager with EMA-adaptive threshold.

    After `warmup_steps` steps the threshold is set to:
      min(ema_usage * (1 + safety_margin), gpu_capacity_gb * 0.92)
    """

    def __init__(self, threshold_gb: float = 7.5,
                 initial_threshold_gb: Optional[float] = None,
                 page_size: int = 1024, sink_size: int = 4,
                 safety_margin: float = 0.15, ema_alpha: float = 0.3,
                 warmup_steps: int = 10, gpu_capacity_gb: float = 15.5):
        # Accept both threshold_gb (clean API) and initial_threshold_gb (legacy)
        threshold_gb = initial_threshold_gb if initial_threshold_gb is not None else threshold_gb
        super().__init__(threshold_gb=threshold_gb,
                         page_size=page_size, sink_size=sink_size)
        self.safety_margin = safety_margin
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps
        self.gpu_capacity_gb = gpu_capacity_gb
        self._ema: Optional[float] = None
        self._step = 0

    def _update_threshold(self, device: torch.device) -> None:
        cur = _memory_allocated_gb(device)
        self._step += 1
        self._ema = cur if self._ema is None else \
            self.ema_alpha * cur + (1 - self.ema_alpha) * self._ema
        if self._step > self.warmup_steps:
            adaptive = min(self._ema * (1 + self.safety_margin),
                           self.gpu_capacity_gb * 0.92)
            self.threshold_bytes = adaptive * 1024 ** 3

    def offload_kv_cache(self, cache, device: Optional[torch.device] = None):
        dev = device or _infer_device()
        self._update_threshold(dev)
        return super().offload_kv_cache(cache, dev)

    def reset(self) -> None:
        super().reset()
        self._ema = None
        self._step = 0


class PredictiveMemoryPolicy(AdaptiveVirtualPageManager):
    """
    AdaptiveVirtualPageManager that predicts VRAM growth from model architecture.

    Call `calibrate_for_model(model)` after loading any HuggingFace model to
    automatically infer num_kv_heads, head_dim, num_layers, and sliding-window
    layer layout from model.config.

    Compatible with:
      - Mistral, Llama-3, Phi, Qwen (full attention, transformers >= 4.46)
      - Gemma 4 E2B/E4B/27B (sliding-window hybrid, transformers >= 5.5.0)
      - Any HuggingFace AutoModelForCausalLM with standard config attributes

    Sliding-window awareness (Gemma 3 / Gemma 4 / Mistral-Sliding):
      Local (sliding-window) layers already cap their KV at `sliding_window`
      tokens; cognito skips eviction on those layers to avoid corrupting the
      window state. Only global (full-attention) layers are eviction targets.
    """

    # Safe defaults (Mistral-7B layout)
    _num_kv_heads: int = 8
    _head_dim: int = 128
    _num_layers: int = 32
    _bytes_per_elem: int = 2   # fp16
    _kv_factor: int = 2        # keys + values
    # Set of layer indices that use sliding-window attention (never evicted).
    _sw_layers: frozenset = frozenset()
    _sliding_window_size: int = 0   # 0 = no sliding window

    def calibrate_for_model(self, model) -> None:
        """
        Read architecture parameters from model.config.
        Handles standard (MHA/GQA) and hybrid sliding-window models (Gemma 4).
        """
        cfg = model.config
        self._num_kv_heads = getattr(cfg, "num_key_value_heads",
                              getattr(cfg, "num_attention_heads", 8))
        n_heads = getattr(cfg, "num_attention_heads", self._num_kv_heads)
        self._head_dim = getattr(cfg, "head_dim",
                          getattr(cfg, "hidden_size", 4096) // max(1, n_heads))
        self._num_layers = getattr(cfg, "num_hidden_layers", 32)

        # Detect element size from model dtype
        try:
            p = next(model.parameters())
            self._bytes_per_elem = (4 if p.dtype == torch.float32 else
                                    2 if p.dtype in (torch.bfloat16, torch.float16)
                                    else 1)
        except StopIteration:
            pass

        # ── Sliding-window layer detection (Gemma 3 / Gemma 4 / Mistral-SW) ──
        # Gemma 4 config exposes `layer_types` list: 'sliding_attention' | 'full_attention'
        # Gemma 3 uses `sliding_window` + alternating pattern (every Nth layer is global).
        # Mistral-Sliding: sliding_window attribute, uniform across all layers.
        sw = getattr(cfg, "sliding_window", None) or getattr(cfg, "sliding_window_size", None)
        self._sliding_window_size = int(sw) if sw else 0
        sw_layers: set = set()

        if self._sliding_window_size:
            layer_types = getattr(cfg, "layer_types", None)
            if layer_types and isinstance(layer_types, (list, tuple)):
                # Gemma 4 style: explicit per-layer type list
                sw_layers = {i for i, t in enumerate(layer_types)
                             if str(t).lower() in ("sliding_attention", "local", "window")}
            else:
                # Gemma 3 / generic: infer from interleave_sliding_window ratio
                period = getattr(cfg, "interleave_sliding_window", None)
                if period and isinstance(period, int) and period > 1:
                    # Every `period`-th layer is global; rest are local/sliding
                    sw_layers = {i for i in range(self._num_layers) if (i + 1) % period != 0}
                elif getattr(cfg, "attention_bias", None) is None:
                    # Conservative fallback: mark nothing, evict uniformly
                    pass
            if sw_layers:
                n_global = self._num_layers - len(sw_layers)
                print(f"[Cognito] Sliding-window model detected: "
                      f"sw={self._sliding_window_size} tok, "
                      f"{len(sw_layers)} local layers, {n_global} global layers.")
        self._sw_layers = frozenset(sw_layers)

    def predict_kv_bytes(self, delta_tokens: int) -> float:
        """
        Estimate bytes added to KV cache by delta_tokens new tokens.
        Accounts for sliding-window layers that cap contribution at sw_size.
        """
        if self._sw_layers and self._sliding_window_size:
            n_global = self._num_layers - len(self._sw_layers)
            n_local  = len(self._sw_layers)
            # Local layers are already at sw_size capacity after warmup
            global_bytes = (delta_tokens * self._num_kv_heads * self._head_dim
                            * n_global * self._kv_factor * self._bytes_per_elem)
            # Local layers contribute only if below sw_size
            local_bytes  = (min(delta_tokens, self._sliding_window_size)
                            * self._num_kv_heads * self._head_dim
                            * n_local * self._kv_factor * self._bytes_per_elem)
            return global_bytes + local_bytes
        return (delta_tokens * self._num_kv_heads * self._head_dim
                * self._num_layers * self._kv_factor * self._bytes_per_elem)

    def should_evict(self, device: torch.device, delta_tokens: int = 1) -> bool:
        cur = _memory_allocated_gb(device)
        extra = self.predict_kv_bytes(delta_tokens) / 1024 ** 3
        return (cur + extra) * 1024 ** 3 > self.threshold_bytes

    def offload_kv_cache(self, cache, device: Optional[torch.device] = None):
        if not self.active or cache is None:
            return cache
        dev = device or _infer_device()
        if not self.should_evict(dev):
            return cache
        return VirtualPageManager.offload_kv_cache(self, cache, dev)


class RAGAwarePager(PredictiveMemoryPolicy):
    """
    Segment-aware KV cache eviction by retrieval-relevance score.

    Eviction is restricted to the pre-decode boundary so that absolute
    RoPE positions established during prefill are never invalidated
    (RoPE drift fix — original contribution, not in published literature).

    Usage:
        pager = RAGAwarePager(threshold_gb=5.5)
        engine = CognitoEngine(model, tokenizer, pager=pager)

        result = engine.chat(
            messages=[{"role": "user", "content": query}],
            segments=[
                ContextSegment(text=passage, score=rrf_score),
                ...
            ],
        )

    The engine registers each ContextSegment via register_segment_abs().
    You can also call register_segment_abs() directly for custom pipelines.
    """

    class _Segment:
        __slots__ = ("seg_id", "score", "start", "end", "label", "turn")

        def __init__(self, seg_id, score, start, end, label, turn):
            self.seg_id = seg_id
            self.score = score
            self.start = start
            self.end = end
            self.label = label
            self.turn = turn

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._segments: list = []
        self._reserved: list = []   # [(start, end, label)]
        self._next_id = 0
        self._evictions = 0
        self.global_turn = 0

    # ── Registration ────────────────────────────────────────────────────────

    def reserve_range(self, start: int, end: int, label: str = "") -> None:
        """Mark token span [start, end) as never-evictable (e.g. system prompt)."""
        if end > start:
            self._reserved.append((start, end, label))

    def register_segment_abs(self, start: int, end: int, score: float,
                              label: str = "", turn: int = -1) -> int:
        """
        Register a passage by its absolute token positions in the prefilled cache.
        Returns the segment id. Raises if the span overlaps a reserved range.
        """
        if end <= start:
            raise ValueError(f"Invalid span [{start}, {end})")
        for rs, re, rl in self._reserved:
            if start < re and end > rs:
                raise ValueError(
                    f"Segment '{label}' [{start},{end}) overlaps reserved "
                    f"range '{rl}' [{rs},{re})")
        seg = RAGAwarePager._Segment(
            seg_id=self._next_id, score=score,
            start=start, end=end, label=label,
            turn=self.global_turn if turn < 0 else turn)
        self._segments.append(seg)
        self._next_id += 1
        return seg.seg_id

    # ── Eviction ────────────────────────────────────────────────────────────

    def maybe_evict_pre_decode(self, cache, device: Optional[torch.device] = None):
        """
        One-shot pre-decode eviction: drop the lowest-relevance segment if
        VRAM is above threshold. Call once between prefill and decode.
        """
        if not self._segments or cache is None:
            return cache
        dev = device or _infer_device()
        if not self.should_evict(dev):
            return cache

        def _score(s: RAGAwarePager._Segment) -> float:
            lag = self.global_turn - s.turn
            return s.score * math.exp(-0.4 * lag)

        self._segments.sort(key=_score)
        victim = self._segments.pop(0)
        return self._evict_segment(cache, victim, dev)

    def _evict_segment(self, cache, victim: "_Segment", device: torch.device):
        start, end = victim.start, victim.end
        keys, vals = _get_kv_tensors(cache)
        new_k, new_v = [], []
        seq = keys[0].shape[-2]
        s, e = min(start, seq), min(end, seq)
        removed = e - s

        for layer_idx, (k, v) in enumerate(zip(keys, vals)):
            # Skip sliding-window layers: they cap their own KV internally;
            # slicing them would corrupt the window state.
            if layer_idx in self._sw_layers:
                new_k.append(k); new_v.append(v); continue
            cur_seq = k.shape[-2]
            cs, ce = min(start, cur_seq), min(end, cur_seq)
            if cs >= ce:
                new_k.append(k); new_v.append(v); continue
            if cs == 0:
                nk = k[:, :, ce:, :]
                nv = v[:, :, ce:, :]
            elif ce >= cur_seq:
                nk = k[:, :, :cs, :]
                nv = v[:, :, :cs, :]
            else:
                nk = torch.cat([k[:, :, :cs, :], k[:, :, ce:, :]], dim=2)
                nv = torch.cat([v[:, :, :cs, :], v[:, :, ce:, :]], dim=2)
            new_k.append(nk.contiguous())
            new_v.append(nv.contiguous())

        # Shift surviving segment positions
        kept = []
        for seg in self._segments:
            if seg.start >= e:
                seg.start -= removed; seg.end -= removed
            elif seg.end > s:
                seg.end = max(seg.start, seg.end - removed)
            if seg.end > seg.start:
                kept.append(seg)
        self._segments = kept

        # Shift reserved ranges
        new_reserved = []
        for rs, re, rl in self._reserved:
            if rs >= e:
                new_reserved.append((rs - removed, re - removed, rl))
            elif re > s:
                new_reserved.append((rs, max(rs, re - removed), rl))
            else:
                new_reserved.append((rs, re, rl))
        self._reserved = new_reserved

        self._evictions += 1
        new_cache = _rebuild_cache(new_k, new_v)
        _sync(device); _empty_cache(device)

        if self._evictions <= 5:
            print(f"[Cognito/RAGAware] evict #{self._evictions} "
                  f"'{victim.label}' score={victim.score:.3f} (-{removed} tok)")
        return new_cache

    # ── Overrides ────────────────────────────────────────────────────────────

    def offload_kv_cache(self, cache, device: Optional[torch.device] = None):
        # During decode: only StreamingLLM sink/window offload.
        # Segment-aware eviction is restricted to pre-decode (RoPE drift fix).
        if not self.active or cache is None:
            return cache
        dev = device or _infer_device()
        if not self.should_evict(dev):
            return cache
        return VirtualPageManager.offload_kv_cache(self, cache, dev)

    def reset(self) -> None:
        super().reset()
        self._segments.clear()
        self._reserved.clear()
        self._next_id = 0
        self._evictions = 0

    @property
    def eviction_count(self) -> int:
        return self._evictions


class H2OEvictionPolicy:
    """
    H2O Heavy-Hitter Oracle (Zhang et al., NeurIPS 2023).

    Uses L2-norm inverse of key vectors as an attention-mass proxy
    (Devoto et al., EMNLP 2024), which is compatible with SDPA and NF4
    without requiring output_attentions=True.

    Keeps: sink_size attention-sink tokens + heavy_ratio high-importance
    tokens + recent_ratio most-recent tokens. Evicts the rest.

    Sliding-window aware: layers already bounded by a sliding window are
    skipped to avoid corrupting the window state (required for Gemma 4).
    """

    def __init__(self, threshold_gb: float = 5.5, gpu_capacity_gb: float = 16.0,
                 heavy_ratio: float = 0.30, recent_ratio: float = 0.10,
                 sink_size: int = 4):
        self.threshold_bytes = threshold_gb * 1024 ** 3
        self.gpu_capacity_gb = gpu_capacity_gb
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.sink_size = sink_size
        self.active = False
        self._evictions = 0
        self._sw_layers: frozenset = frozenset()

    def reset(self) -> None:
        self._evictions = 0

    @property
    def eviction_count(self) -> int:
        return self._evictions

    @property
    def streamllm_offloads(self) -> int:
        return 0

    @property
    def blocks_in_cpu(self) -> int:
        return 0

    def calibrate_for_model(self, model) -> None:
        """Detect sliding-window layers so H2O skips them during eviction."""
        cfg = model.config
        sw = getattr(cfg, "sliding_window", None) or getattr(cfg, "sliding_window_size", None)
        if not sw:
            return
        n_layers = getattr(cfg, "num_hidden_layers", 0)
        layer_types = getattr(cfg, "layer_types", None)
        sw_layers: set = set()
        if layer_types and isinstance(layer_types, (list, tuple)):
            sw_layers = {i for i, t in enumerate(layer_types)
                         if str(t).lower() in ("sliding_attention", "local", "window")}
        else:
            period = getattr(cfg, "interleave_sliding_window", None)
            if period and isinstance(period, int) and period > 1:
                sw_layers = {i for i in range(n_layers) if (i + 1) % period != 0}
        self._sw_layers = frozenset(sw_layers)

    def _should_evict(self, device: torch.device) -> bool:
        return _memory_allocated_gb(device) * 1024 ** 3 > self.threshold_bytes

    def _evict_layer(self, k: torch.Tensor, v: torch.Tensor,
                     heavy_budget: int, recent_budget: int
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = k.shape[2]
        keep = self.sink_size + heavy_budget + recent_budget
        if keep >= seq:
            return k, v

        # L2-norm inverse: lower norm keys attract more attention (Devoto 2024)
        importance = 1.0 / torch.norm(k, p=2, dim=-1).clamp(min=1e-8)
        importance = importance.mean(dim=1)  # [batch, seq]

        nk_list, nv_list = [], []
        for b in range(k.shape[0]):
            sc = importance[b]
            sink_idx = torch.arange(self.sink_size, device=k.device)
            recent_idx = torch.arange(
                max(self.sink_size, seq - recent_budget), seq, device=k.device)
            mask = torch.ones(seq, dtype=torch.bool, device=k.device)
            mask[:self.sink_size] = False
            mask[recent_idx] = False
            cand_sc = sc.clone()
            cand_sc[~mask] = -1.0
            n_heavy = min(heavy_budget, int(mask.sum()))
            _, heavy_idx = cand_sc.topk(k=max(1, n_heavy), dim=-1)
            keep_idx, _ = torch.unique(
                torch.cat([sink_idx, heavy_idx, recent_idx])).sort()
            nk_list.append(k[b:b+1, :, keep_idx, :])
            nv_list.append(v[b:b+1, :, keep_idx, :])
        return torch.cat(nk_list, dim=0), torch.cat(nv_list, dim=0)

    def maybe_evict_pre_decode(self, cache, device: Optional[torch.device] = None):
        """Apply H2O eviction to all layers. Call once before decoding."""
        if cache is None:
            return cache
        dev = device or _infer_device()
        if not self._should_evict(dev):
            return cache
        try:
            keys, vals = _get_kv_tensors(cache)
        except Exception:
            return cache

        seq = keys[0].shape[2]
        heavy = max(self.sink_size, int(self.heavy_ratio * seq))
        recent = max(self.sink_size, int(self.recent_ratio * seq))

        new_k, new_v, evicted = [], [], 0
        for layer_idx, (k, v) in enumerate(zip(keys, vals)):
            if layer_idx in self._sw_layers:
                new_k.append(k)
                new_v.append(v)
                continue
            nk, nv = self._evict_layer(k, v, heavy, recent)
            evicted = max(evicted, k.shape[2] - nk.shape[2])
            new_k.append(nk)
            new_v.append(nv)

        if evicted > 0:
            self._evictions += 1
            _sync(dev); _empty_cache(dev)
            if self._evictions <= 5:
                print(f"[Cognito/H2O] evict #{self._evictions}: "
                      f"{seq} -> {new_k[0].shape[2]} tok "
                      f"(heavy={heavy} recent={recent} sink={self.sink_size})")
        return _rebuild_cache(new_k, new_v)

    def offload_kv_cache(self, cache, device=None):
        return cache  # H2O only runs pre-decode


# ─── Public data types ────────────────────────────────────────────────────────

@dataclass
class ContextSegment:
    """
    A passage or text segment to be registered in the KV cache pager.

    Attributes:
        text     — raw text of the segment (used for span detection)
        score    — retrieval relevance score (higher = more important, evict last)
        label    — optional debug label (e.g. "passage_1")
        reserved — if True, this segment is never evicted (e.g. system prompt)
    """
    text: str
    score: float = 1.0
    label: str = ""
    reserved: bool = False


@dataclass
class GenerationResult:
    """Output of CognitoEngine.generate() / .chat()."""
    text: str
    input_tokens: int
    output_tokens: int
    ttft_s: float        # time-to-first-token (seconds)
    itl_ms: float        # inter-token latency (ms), average over decode steps
    peak_vram_gb: float
    evictions: int       # number of segment evictions performed
    offloads: int        # number of StreamingLLM offload events
    status: str          # "ok" | "oom"


# ─── Core engine ─────────────────────────────────────────────────────────────

class CognitoEngine:
    """
    Application-level KV cache paging engine for HuggingFace models.

    Compatible with any AutoModelForCausalLM on CUDA, MPS, or CPU.
    Supports RAGAwarePager, H2OEvictionPolicy, and VirtualPageManager.

    Parameters:
        model             — loaded HuggingFace model (any quantization)
        tokenizer         — corresponding AutoTokenizer
        pager             — eviction policy instance, or None (no eviction)
        chunk_size        — tokens per prefill chunk (default 256, cf. Sarathi-Serve)
        min_chunked_tokens — minimum prompt length to trigger chunked prefill
        system_prompt     — injected into chat() messages as a system turn
    """

    def __init__(self, model, tokenizer, pager=None,
                 chunk_size: int = 256, min_chunked_tokens: int = 512,
                 system_prompt: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.pager = pager
        self.chunk_size = chunk_size
        self.min_chunked_tokens = min_chunked_tokens
        self.system_prompt = system_prompt
        self._device = _infer_device(model)

        # Calibrate pager to this model's architecture
        if pager is not None and hasattr(pager, "calibrate_for_model"):
            pager.calibrate_for_model(model)

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        segments: Optional[List[ContextSegment]] = None,
        max_new_tokens: int = 200,
        use_chunked_prefill: bool = True,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """
        Generate text from a pre-formatted prompt string.

        If `segments` is provided and the pager is a RAGAwarePager, the engine
        locates each segment's token span within the prompt and registers it
        for segment-aware eviction.

        Args:
            prompt            — complete formatted prompt (use chat() for
                                automatic chat-template application)
            segments          — optional list of ContextSegment for eviction tracking
            max_new_tokens    — maximum tokens to generate
            use_chunked_prefill — enable Sarathi-Serve chunked prefill
            temperature       — 0.0 = greedy, >0 = sampling
        """
        input_ids, spans = self._tokenize_with_spans(prompt, segments)
        return self._run(input_ids, spans, segments, max_new_tokens,
                         use_chunked_prefill, temperature)

    def chat(
        self,
        messages: List[dict],
        segments: Optional[List[ContextSegment]] = None,
        max_new_tokens: int = 200,
        use_chunked_prefill: bool = True,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """
        Generate from a list of chat messages, injecting context segments into
        the last user turn. Applies the tokenizer's chat template automatically.

        Args:
            messages  — list of {"role": ..., "content": ...} dicts
            segments  — optional context passages for eviction tracking
            ...       — same as generate()

        Example:
            result = engine.chat(
                messages=[{"role": "user", "content": "Who wrote Hamlet?"}],
                segments=[
                    ContextSegment("Shakespeare wrote Hamlet around 1600.", score=0.9),
                    ContextSegment("It is set in Denmark.", score=0.3),
                ],
            )
        """
        msgs = list(messages)

        # Optionally prepend system message
        if self.system_prompt and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": self.system_prompt}] + msgs

        # Inject context into the last user message
        if segments:
            ctx_parts = []
            for i, seg in enumerate(segments):
                ctx_parts.append(f"[{i+1}] {seg.text}")
            ctx_block = "Context:\n" + "\n\n".join(ctx_parts)
            for i in reversed(range(len(msgs))):
                if msgs[i]["role"] == "user":
                    msgs[i] = dict(msgs[i],
                                   content=f"{ctx_block}\n\nQuestion: {msgs[i]['content']}")
                    break

        prompt = self._apply_template(msgs)
        return self.generate(prompt, segments=segments,
                             max_new_tokens=max_new_tokens,
                             use_chunked_prefill=use_chunked_prefill,
                             temperature=temperature)

    def chunked_prefill(self, input_ids: torch.Tensor,
                        past_kv=None) -> Tuple[object, int]:
        """
        Low-level chunked prefill (Sarathi-Serve OSDI 2024 style).

        Processes input_ids in chunks of self.chunk_size tokens, accumulating
        the KV cache via explicit cache_position. No eviction is performed here.

        Returns:
            (past_key_values, current_seq_len)
        """
        return self._chunked_prefill(input_ids, past_kv)

    def evict_pre_decode(self, past_kv,
                         device: Optional[torch.device] = None) -> object:
        """
        Apply pre-decode eviction with the configured pager.
        For RAGAwarePager: segment-aware eviction.
        For H2OEvictionPolicy: L2-norm based eviction.
        """
        dev = device or self._device
        if isinstance(self.pager, (RAGAwarePager, H2OEvictionPolicy)):
            return self.pager.maybe_evict_pre_decode(past_kv, dev)
        return past_kv

    # ── Internal: tokenization ─────────────────────────────────────────────

    def _apply_template(self, messages: List[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        # Generic fallback
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _tokenize_with_spans(self, prompt: str,
                              segments: Optional[List[ContextSegment]]
                              ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """
        Tokenize prompt and optionally locate each segment's token span.
        Returns (input_ids [1, T], spans_or_None).

        Span detection uses return_offsets_mapping when the tokenizer supports
        it (most fast tokenizers). Falls back to no span tracking otherwise.
        """
        try:
            enc = self.tokenizer(prompt, return_tensors="pt",
                                  return_offsets_mapping=True,
                                  add_special_tokens=False)
            input_ids = enc["input_ids"].to(self._device)
            offsets = enc["offset_mapping"][0].tolist()  # [(char_s, char_e), ...]

            spans = None
            if segments and isinstance(self.pager, RAGAwarePager):
                spans = self._locate_spans(prompt, offsets, segments)
            return input_ids, spans

        except Exception:
            enc = self.tokenizer(prompt, return_tensors="pt",
                                  add_special_tokens=False)
            return enc["input_ids"].to(self._device), None

    def _locate_spans(self, prompt: str, offsets: List[Tuple[int, int]],
                      segments: List[ContextSegment]) -> List[Tuple[int, int, float, str, bool]]:
        """
        Map each segment's text to (tok_start, tok_end, score, label, reserved)
        using character offset mapping from the tokenizer.
        """
        result = []
        search_from = 0
        for seg in segments:
            char_s = prompt.find(seg.text, search_from)
            if char_s < 0:
                # Try from beginning (segment text might appear earlier due to dedup)
                char_s = prompt.find(seg.text)
            if char_s < 0:
                warnings.warn(f"[Cognito] Segment '{seg.label}' not found in prompt; skipping.")
                continue
            char_e = char_s + len(seg.text)
            search_from = char_e

            tok_s = next((i for i, (cs, ce) in enumerate(offsets) if cs >= char_s), None)
            tok_e = next((i for i, (cs, ce) in reversed(list(enumerate(offsets)))
                          if ce <= char_e), None)
            if tok_s is None or tok_e is None or tok_e <= tok_s:
                continue
            result.append((tok_s, tok_e + 1, seg.score, seg.label, seg.reserved))
        return result

    # ── Internal: prefill & decode ─────────────────────────────────────────

    def _chunked_prefill(self, input_ids: torch.Tensor, past_kv=None
                          ) -> Tuple[object, int]:
        n_total = input_ids.shape[-1]
        cur_len = 0 if past_kv is None else get_cache_seq_len(past_kv)
        with torch.no_grad():
            for start in range(0, n_total, self.chunk_size):
                end = min(start + self.chunk_size, n_total)
                chunk = input_ids[:, start:end]
                chunk_n = chunk.shape[-1]
                mask = torch.ones((1, cur_len + chunk_n),
                                   dtype=torch.long, device=self._device)
                cache_pos = torch.arange(cur_len, cur_len + chunk_n,
                                          device=self._device)
                out = self.model(input_ids=chunk, attention_mask=mask,
                                  past_key_values=past_kv,
                                  cache_position=cache_pos, use_cache=True)
                past_kv = out.past_key_values
                cur_len += chunk_n
        return past_kv, cur_len

    def _decode_loop(self, input_ids: torch.Tensor, max_new_tokens: int,
                      past_kv, cur_len: int,
                      temperature: float, t0: float
                      ) -> Tuple[torch.Tensor, float, float]:
        """
        Autoregressive decode loop. Returns (generated_ids, ttft_s, itl_ms).
        `input_ids` should be the single "trigger" token (last token of prompt).
        """
        generated = input_ids.clone()
        ttft = 0.0
        itl_samples: List[float] = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                ts = time.perf_counter()
                model_in = generated[:, -1:] if (step > 0 or past_kv is not None) \
                           else generated
                cur_n = model_in.shape[-1]
                mask = torch.ones((1, cur_len + cur_n),
                                   dtype=torch.long, device=self._device)
                cache_pos = torch.arange(cur_len, cur_len + cur_n,
                                          device=self._device)

                out = self.model(input_ids=model_in, attention_mask=mask,
                                  past_key_values=past_kv,
                                  cache_position=cache_pos, use_cache=True)

                logits = out.logits[:, -1, :]
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                else:
                    next_tok = logits.argmax(dim=-1, keepdim=True)

                past_kv = out.past_key_values
                cur_len += cur_n

                # In-decode eviction removed: modifying KV positions during
                # autoregressive generation causes RoPE drift. Eviction is
                # applied only pre-decode (see _run).
                if self.pager is not None and self.pager.active:
                    past_kv = self.pager.offload_kv_cache(past_kv, self._device)

                generated = torch.cat([generated, next_tok], dim=-1)
                te = time.perf_counter()
                if step == 0:
                    ttft = te - t0
                else:
                    itl_samples.append((te - ts) * 1000)

                if next_tok.item() == self.tokenizer.eos_token_id:
                    break

        itl = sum(itl_samples) / len(itl_samples) if itl_samples else 0.0
        return generated, ttft, itl

    def _run(self, input_ids: torch.Tensor,
             spans: Optional[List[Tuple]],
             segments: Optional[List[ContextSegment]],
             max_new_tokens: int, use_chunked_prefill: bool,
             temperature: float) -> GenerationResult:

        n_input = input_ids.shape[-1]
        dev = self._device
        _reset_peak(dev)
        t0 = time.perf_counter()

        # Reset / activate pager
        if self.pager is not None:
            self.pager.reset()
            self.pager.active = True

        try:
            # ── Prefill ─────────────────────────────────────────────────
            if use_chunked_prefill and n_input >= self.min_chunked_tokens:
                # Process all-but-last token in chunked prefill
                past_kv, cur_len = self._chunked_prefill(input_ids[:, :-1])

                # Register segments with exact token spans
                if spans and isinstance(self.pager, RAGAwarePager):
                    self._register_spans(spans, cur_len)

                # Pre-decode eviction (RoPE-safe window)
                past_kv = self.evict_pre_decode(past_kv, dev)
                cur_len = get_cache_seq_len(past_kv)

                # Decode starts from last input token
                trigger = input_ids[:, -1:]
                gen, ttft, itl = self._decode_loop(
                    trigger, max_new_tokens, past_kv, cur_len, temperature, t0)
                # Reconstruct full output ids for clean decoding
                output_ids = gen
                n_output = gen.shape[-1] - 1  # trigger + new tokens - trigger

            else:
                # Short prompt: standard model.generate() fast path
                if self.pager is not None:
                    self.pager.active = False
                with torch.no_grad():
                    gen = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=max_new_tokens,
                        do_sample=(temperature > 0),
                        temperature=temperature if temperature > 0 else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                ttft = time.perf_counter() - t0
                itl = 0.0
                output_ids = gen[:, n_input:]
                n_output = output_ids.shape[-1]

            status = "ok"

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                _empty_cache(dev)
                return GenerationResult(
                    text="", input_tokens=n_input, output_tokens=0,
                    ttft_s=0.0, itl_ms=0.0,
                    peak_vram_gb=_memory_peak_gb(dev),
                    evictions=getattr(self.pager, "eviction_count", 0),
                    offloads=getattr(self.pager, "streamllm_offloads", 0),
                    status="oom")
            raise

        # Decode output text
        if use_chunked_prefill and n_input >= self.min_chunked_tokens:
            # output_ids here is [trigger + new_tokens], skip trigger
            text = self.tokenizer.decode(output_ids[0, 1:],
                                          skip_special_tokens=True).strip()
        else:
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return GenerationResult(
            text=text,
            input_tokens=n_input,
            output_tokens=n_output,
            ttft_s=ttft,
            itl_ms=itl,
            peak_vram_gb=_memory_peak_gb(dev),
            evictions=getattr(self.pager, "eviction_count", 0),
            offloads=getattr(self.pager, "streamllm_offloads", 0),
            status=status,
        )

    def _register_spans(self, spans: List[Tuple], cur_len: int) -> None:
        """Register located token spans with the RAGAwarePager."""
        pager = self.pager
        for tok_s, tok_e, score, label, reserved in spans:
            if tok_e > cur_len:
                continue  # span beyond what was prefilled
            if reserved:
                pager.reserve_range(tok_s, tok_e, label=label)
            else:
                try:
                    pager.register_segment_abs(tok_s, tok_e, score=score, label=label)
                except (ValueError, AssertionError) as exc:
                    warnings.warn(f"[Cognito] Could not register segment "
                                  f"'{label}': {exc}")


# ─── Convenience: model loading helper ───────────────────────────────────────

def load_model(model_name: str, quantization: str = "nf4",
               device_map: str = "auto", attn_impl: str = "sdpa"):
    """
    Load a HuggingFace causal LM with optional NF4/int8 quantization.

    Args:
        model_name   — HuggingFace model id or local path
        quantization — "nf4" | "int8" | "none"
        device_map   — "auto" or a specific device string
        attn_impl    — "sdpa" (default) | "eager" | "flash_attention_2"

    Returns:
        (model, tokenizer)

    Example:
        model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.3")
        engine = CognitoEngine(model, tokenizer, pager=RAGAwarePager())
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"device_map": device_map, "trust_remote_code": True}

    if quantization == "nf4":
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["torch_dtype"] = torch.float16
        except ImportError:
            warnings.warn("[Cognito] bitsandbytes not available; loading in fp16.")
            kwargs["torch_dtype"] = torch.float16
    elif quantization == "int8":
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            warnings.warn("[Cognito] bitsandbytes not available; loading in fp16.")
            kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16

    try:
        kwargs["attn_implementation"] = attn_impl
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model.eval()
    return model, tokenizer


# ─── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    print(f"cognito {__version__} — smoke test")
    print(f"PyTorch {torch.__version__} | device: ", end="")

    dev = _infer_device()
    print(dev)

    if "--load" in sys.argv:
        idx = sys.argv.index("--load") + 1
        model_name = sys.argv[idx] if idx < len(sys.argv) else "gpt2"
        quant = "none"
        for a in sys.argv:
            if a.startswith("--quant="):
                quant = a.split("=", 1)[1]

        print(f"Loading {model_name!r} (quant={quant}) ...")
        model, tok = load_model(model_name, quantization=quant)

        pager = RAGAwarePager(threshold_gb=4.0)
        engine = CognitoEngine(model, tok, pager=pager, chunk_size=64,
                                system_prompt="You are a helpful assistant.")

        result = engine.chat(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            segments=[
                ContextSegment("Paris is the capital and most populous city of France.",
                               score=0.95, label="passage_0"),
                ContextSegment("France is a country in Western Europe.",
                               score=0.40, label="passage_1"),
            ],
            max_new_tokens=50,
        )
        print(f"\nResponse : {result.text!r}")
        print(f"Tokens   : {result.input_tokens} in / {result.output_tokens} out")
        print(f"TTFT     : {result.ttft_s*1000:.1f} ms")
        print(f"ITL      : {result.itl_ms:.1f} ms")
        print(f"VRAM     : {result.peak_vram_gb:.2f} GB")
        print(f"Evictions: {result.evictions}")
        print(f"Status   : {result.status}")
    else:
        print("Pass --load <model_id> [--quant=nf4|int8|none] to run a live test.")
        print("Example: python cognito.py --load gpt2 --quant=none")
