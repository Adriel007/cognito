# Cognito: KV Cache Paging em Nível de Aplicação para Long-Context LLM Inference

**Trabalho de Conclusão de Curso — Adriel Souza Andrade**

> *"Application-level KV cache paging com segment awareness preserva qualidade em long-context inference com menor VRAM que sem paginação, em GPUs commodity (T4 16GB)."*

---

## Resumo

O **Cognito** é um sistema de gerenciamento de KV cache para inferência de LLMs em GPUs commodity, implementado inteiramente em userland HuggingFace Transformers — sem CUDA custom, sem fork de runtime. Combina **chunked prefill** (Sarathi-Serve, OSDI 2024) com **evicção segment-aware** por relevância de retrieval (RAGAwarePager), posicionando-se como alternativa prática ao vLLM/PagedAttention para cenários com restrição de VRAM.

---

## Resultados Experimentais

### Tabela Principal — EM (Mistral-7B-Instruct-v0.3 NF4 | T4 16GB)

| Config | NIAH 4k | NIAH 8k | SQuAD 8k | Stress 16/32k | VRAM pico² | ITL³ |
|--------|---------|---------|----------|----------------|-----------|------|
| **L1: NoPaging** | 100%¹ | OOM | OOM | OOM | 10.0 GB / OOM | — |
| **L2: ChunkedPrefill** | 100% | 100% | 73.3% | 50% | 5.11–5.95 GB | 109ms |
| **L3: Cognito** | 100% | 100% | 73.3% | 50% | 5.11–5.95 GB | 107ms |
| **L4: H2O (Zhang 2023)** | 100% | 60% ↓ | 40% ↓↓ | 16.7% ↓↓↓ | 5.11–6.34 GB | **90ms** |
| **L5: QRepeat (LitM)** | 100% | 100% | 73.3% | 50% | 5.11–5.96 GB | 108ms |

¹ L1 OOM em queries ≥8k; apenas NIAH 4k completa (10.0 GB).
² Faixa de VRAM pico por comprimento de contexto: 4k=5.11 GB, 8k=5.83 GB, Stress 16/32k=5.95 GB para L2/L3/L5. O chunked prefill limita o pico a O(chunk²) em vez de O(N²) — razão pela qual Stress 16/32k usa apenas 0.12 GB a mais que 8k.
³ ITL médio sobre NIAH 4k+8k. Varia por comprimento: 4k≈95ms, 8k≈122ms, Stress≈128ms (L2/L3/L5). L4 mantém ≈91–94ms em todos os comprimentos graças à redução do KV cache por evicção.

### F1 Token-Level — SQuAD 8k

| Config | EM | F1 | Gap F1−EM |
|--------|----|----|-----------|
| L2: ChunkedPrefill | 73.3% | **90.4%** | +17.1 pp |
| L3: Cognito | 73.3% | **90.4%** | +17.1 pp |
| L4: H2O | 40.0% | **51.8%** | +11.8 pp |
| L5: QRepeat | 73.3% | **88.1%** | +14.8 pp |

O gap F1−EM de 17 pp em L2/L3 indica que o modelo acerta a substância mas nem sempre a formulação canônica exata do SQuAD. Em L4 o gap cai para 12 pp: a evicção H2O degrada mais a precisão lexical do que a cobertura semântica.

### Paired Bootstrap L3 vs L4 (H0: ΔEM=0, dois lados, n=2000)

| Benchmark | ΔEM (L3−L4) | IC 95% | p | Bonferroni α/4 |
|-----------|-------------|--------|---|----------------|
| NIAH 4k/8k | +20.0 pp | [+6.7, +36.7] | 0.010 | ✓ |
| SQuAD 8k | +33.3 pp | [+13.3, +60.0] | 0.007 | ✓ |
| Stress 16k/32k | +33.3 pp | [+11.1, +55.6] | 0.005 | ✓ |

**L3 = L2** em todos os benchmarks (Δ=0, p=1.0) — pager não degrada qualidade.
**L3 > L4** nos 3 benchmarks com p < 0.0125 Bonferroni — segment-aware supera heurística de atenção.

---

## Arquitetura

### Configurações Avaliadas

| Label | Descrição | Eviction |
|-------|-----------|----------|
| L1 | Baseline sem paging | Nenhuma |
| L2 | Chunked prefill (Sarathi-Serve) | Nenhuma |
| **L3** | **Cognito: chunked + RAGAwarePager** | **Segment-aware por score RRF** |
| L4 | H2O: chunked + heavy-hitter oracle | L2-norm inversa das keys (Devoto 2024) |
| L5 | L3 + Query Repetition (Liu 2024) | Nenhuma |

### Componentes Principais

**RAGAwarePager** — evicção por relevância de segmento
- Registra spans exatos de cada passagem RAG no KV cache (posição absoluta, sem re-tokenização)
- Evicta o segmento com menor score RRF × decaimento temporal Ebbinghaus: `score × e^{-0.4 × lag}`
- Evicção restrita ao boundary pré-decode para preservar contrato de posição RoPE
- Segmentos reservados (system/question/few-shot) nunca são evictados

**Chunked Prefill** — sem pico de memória O(N²)
- Implementação userland via `cache_position` explícito + `DynamicCache` HuggingFace
- Compatível com SDPA + NF4 sem modificações ao runtime
- `chunk_size=256` tokens (cf. Sarathi-Serve, OSDI 2024)

**H2O Baseline** (real, não stub)
- Aproximação L2-norm inversa das keys como proxy de atenção acumulada (Devoto et al., EMNLP 2024)
- Compatível com SDPA sem `output_attentions=True` e sem NF4 overhead adicional
- Mantém: `sink_size=4` + `heavy_ratio=0.30` + `recent_ratio=0.10`
- Evicção aplicada em `maybe_evict_pre_decode` (antes do decode); o campo `evicts` nos traces reflete o counter *pós-chamada*, que é o comportamento esperado

**Lost-in-the-Middle Mitigations** (Liu et al., 2024)
- Query Repetition (L5): repete a query no final do prompt
- BM25 Context Reorder/Trim: implementado mas incompatível com NIAH sintético (fragmenta needle)

**PredictiveMemoryPolicy** — portabilidade
- `calibrate_for_model(model)` infere `num_kv_heads`, `head_dim`, `num_hidden_layers` do `model.config`
- Funciona com Mistral, Llama-3, Gemma, Phi, Qwen e qualquer modelo HF ≥ 4.46

---

## Diagnósticos de Engenharia

### RoPE Drift (P1/P1.5 fix — contribuição original)
Fenômeno descoberto durante o desenvolvimento: offload StreamingLLM *dentro* do chunked prefill cria drift cumulativo entre `cache_position` e posições absolutas das keys RoPE. Resultado: EM=0% antes do fix, EM=100% após restringir evicção ao boundary pré-decode. Não documentado na literatura.

### Piecewise≠Mono Tokenização
Ao dividir contexto em chunks para passage-level prefill, fronteiras de tokenização produzem ±1 token vs tokenização monolítica, desalinhando spans de `register_segment_abs`. Causa falha silenciosa no NIAH (EM=0% com passage_prefill em blob único).

### Chunked Prefill Limita Pico de VRAM em Contextos Longos (observação experimental)
Stress 16k/32k atingiu apenas 5.95 GB de VRAM pico para L2/L3/L5 — 0.12 GB acima do 8k (5.83 GB). O pico de prefill é O(chunk_size²) = O(256²) em vez de O(N²), portanto cresce muito lentamente com N. O KV cache acumulado cresce linearmente, mas seu efeito sobre o pico medido é menor porque a T4 reporta max_memory_allocated no ponto mais alto, que ocorre durante o forward de um único chunk de 256 tokens.

---

## Benchmarks

| Dataset | Tipo | Tokens | n | EM avaliado |
|---------|------|--------|---|-------------|
| NIAH 4k (RULER-style) | Retrieval sintético | ~4k | 15 | Exact Match |
| NIAH 8k (RULER-style) | Retrieval sintético | ~8k | 15 | Exact Match |
| SQuAD + filler NIAH | QA real + contexto expandido | ~8k | 15 | EM + F1 |
| NIAH Stress | Retrieval sintético em contexto longo | 16k/32k | 18 | Exact Match |
| **Total por config** | | | **63** | |

---

## Stack Técnico

| Componente | Versão/Detalhe |
|------------|----------------|
| Modelo | Mistral-7B-Instruct-v0.3, NF4, double quant |
| Atenção | SDPA (`attn_implementation="sdpa"`) |
| Retrieval | ChromaDB + nomic-embed-text-v1.5 + BM25Okapi |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Avaliação | EM SQuAD-canonical, F1 token-level (SQuAD: EM=73.3%, F1=90.4%), paired bootstrap n=2000 |
| Hardware | NVIDIA T4 16GB (Colab) |
| Runtime | `uv` com PEP 723 inline dependencies |

---

## Referências Principais

- Kwon W. et al. (2023). *PagedAttention*. SOSP '23. [2309.06180](https://arxiv.org/abs/2309.06180)
- Agrawal A. et al. (2024). *Sarathi-Serve*. OSDI '24.
- Zhang Z. et al. (2023). *H2O: Heavy-Hitter Oracle*. NeurIPS '23. [2306.14048](https://arxiv.org/abs/2306.14048)
- Xiao G. et al. (2024). *StreamingLLM*. ICLR '24. [2309.17453](https://arxiv.org/abs/2309.17453)
- Liu N.F. et al. (2024). *Lost in the Middle*. TACL. [2307.03172](https://arxiv.org/abs/2307.03172)
- Hsieh C.-P. et al. (2024). *RULER*. arXiv. [2404.06654](https://arxiv.org/abs/2404.06654)
- Devoto A. et al. (2024). *L2 Norm for KV Eviction*. EMNLP '24.
- Li Y. et al. (2024). *SnapKV*. arXiv. [2404.14469](https://arxiv.org/abs/2404.14469)

---

**Status:** Parte prática concluída. Todos os evals rodados. Bootstrap com significância estatística p < 0.0125 (Bonferroni).
