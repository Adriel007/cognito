# Cognito: VRAM Resilience Prototype for Constrained Hardware

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hardware: Nvidia T4 Compatible](https://img.shields.io/badge/Hardware-NVIDIA_T4_16GB-green.svg)]()

Cognito is an experimental engineering prototype designed to test memory resilience and semantic degradation under long-context inference on constrained edge hardware, specifically the 16GB NVIDIA T4.

While cutting-edge inference engines such as vLLM implement true PagedAttention with complex memory segmentation, Cognito explores a simpler, reactive heuristic approach: aggressive Garbage Collection and KV cache clearing triggered by VRAM pressure thresholds. The central goal is to measure **how the system behaves across the scaling path** — not just where it breaks — by combining survival metrics with semantic retention tests.

## Key Engineering Capabilities

*   **Reactive Memory Eviction:** Implements a VRAM pressure heuristic inspired by paging strategies — monitors allocation and forces aggressive Python/PyTorch garbage collection when predefined thresholds are reached.
*   **OOM Fault Prevention:** Prioritizes system stability and continuous operation over maintaining the full historical KV cache, acting as a survival mechanism under extreme context bloat.
*   **Low-Bitwidth Quantization:** Utilizes 4-bit NormalFloat (NF4) quantization to drastically reduce the baseline memory footprint of the model architecture.
*   **Prototyping on Constraints:** Evaluates models like `Qwen/Qwen3.5-0.8B` to establish operational boundaries on generic 16GB GPUs.

## Architectural Design

The engine is encapsulated within the `CognitoEngine` class wrapper, which coordinates:
1. **VRAM Threshold Monitor:** A reactive eviction heuristic (`VirtualPageManager`) that monitors `torch.cuda.memory_reserved`.
2. **Quantized Base Model:** NF4-loaded causal language models via bitsandbytes.
3. **Generation Loop:** Custom generation cycles integrated with active intervention to prevent `RuntimeError: out of memory`.

---

## Evaluation Methodology

The experimental design prioritizes **behavioral analysis over brute-force stress testing**. Instead of scaling context until OOM, the evaluation uses fixed context sizes, standardized retrieval tests, and controlled baselines.

### 1. Hardware Environment
*   **GPU:** NVIDIA Tesla T4 (15.83 GB VRAM available)
*   **CPU:** Intel Xeon Base
*   **Base Model:** `Qwen/Qwen3.5-0.8B` (Quantized NF4)

### 2. Evaluation Axes

| Axis | Description |
| :--- | :--- |
| **Needle-in-a-Haystack** | Embeds a known fact at a **randomized position** (start, middle, end) within diverse, non-repetitive padding. A **distractor needle** with a false value tests semantic discrimination, not just raw recall. |
| **Fixed Context Evaluation** | Tests at `[8k, 16k, 32k]` tokens with **3 repetitions** per size (randomized needle position each trial). |
| **Baseline Comparison** | Same model, quantization, **and GC policy** — only the proactive VRAM threshold check is removed. Isolates the effect of the monitoring heuristic. |
| **Threshold Sensitivity** | Evaluates Cognito under `[12, 13.5]` GB VRAM thresholds to characterize configuration sensitivity. |

### 3. Methodological Controls

*   **Reproducibility**: Fixed seeds (`random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`).
*   **VRAM metrics** are reset via `torch.cuda.reset_peak_memory_stats()` before each generation in both Cognito and baseline.
*   **Baseline fairness**: both Cognito and baseline perform identical GC and VRAM stats reset; only the proactive threshold check differs.
*   **Needle retrieval** uses normalized string matching to handle formatting variations.
*   **Distractor needle** with a different value is embedded at a random position in the padding.
*   **Diverse padding**: 20 unique sentences from different domains (science, history, economics, etc.) prevent model compression artifacts.

### 4. Metrics

| Metric | Definition |
| :--- | :--- |
| **Survival Rate** | % of runs that completed without OOM failure |
| **OOM Rate** | 1 - Survival Rate (direct measure of VRAM resilience) |
| **Retention Accuracy** | % of runs where the model correctly retrieved the real needle (not the distractor) |
| **Retention Drop** | Retention at context X / Retention at 8k (normalized degradation curve) |

### 5. Results

| Context | Method   | Survival | OOM Rate | Retention | Ret. Drop |
| :---    | :---     | :---     | :---     | :---      | :---      |
| 8k      | Baseline | —        | —        | —         | —         |
| 8k      | Cognito  | —        | —        | —         | —         |
| 16k     | Baseline | —        | —        | —         | —         |
| 16k     | Cognito  | —        | —        | —         | —         |
| 32k     | Baseline | —        | —        | —         | —         |
| 32k     | Cognito  | —        | —        | —         | —         |

*Results to be populated after benchmark execution.*

### 6. Threshold Sensitivity

| Context | 12 GB Survival | 12 GB Retention | 13.5 GB Survival | 13.5 GB Retention |
| :---    | :---           | :---            | :---             | :---              |
| 8k      | —              | —               | —                | —                 |
| 16k     | —              | —               | —                | —                 |
| 32k     | —              | —               | —                | —                 |

*Results to be populated after benchmark execution.*

### 7. Retention by Needle Position

| Position | Baseline Retention | Cognito Retention |
| :---     | :---               | :---              |
| Start    | —                  | —                 |
| Middle   | —                  | —                 |
| End      | —                  | —                 |

*Aggregated across all context sizes. Results to be populated after benchmark execution.*

---

## Limitations & Future Work

*   **Attention Dilution:** Because the engine flushes memory to survive, semantic retention is expected to degrade at higher context sizes. The Needle-in-a-Haystack tests with distractor quantify this degradation explicitly.
*   **True Paging:** The current system uses heuristic cache clearing, not true block-level PagedAttention (like vLLM). A comparison against vLLM's PagedAttention on the same hardware would strengthen the evaluation.
*   **Positional Breakdown:** While needle position is randomized, a dedicated per-position-per-context analysis would provide finer-grained attention profiling with more repetitions.

## Contributing and License

Cognito is an ongoing engineering initiative. Collaborative research addressing KV cache optimization algorithms and parameter-efficient model evaluations is encouraged.

Released under the MIT License.
