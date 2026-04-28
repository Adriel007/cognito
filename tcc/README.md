# Cognito: Sistema de Inferência LLM com Paging Dinâmico de KV Cache

**Trabalho de Conclusão de Curso (TCC) — Documentação Técnica e Científica**

O **Cognito** é uma arquitetura *State-of-the-Art* para gerenciamento ativo do KV Cache de Grandes Modelos de Linguagem (LLMs) em cenários de extrema restrição de hardware (Edge Computing / GPUs Limitadas). Em vez de delegar a memória à lentidão da RAM via paginação cega e agnóstica de sistema operacional (PCIe offload), o Cognito converte a limitação de VRAM em um filtro cognitivo ativo: evictando seletivamente ruído semântico ou passagens de longo-prazo decaídas, garantindo inferência segura, alta velocidade de geração e zero perda de integridade relacional.

---

## 1. Especificações Técnicas Centrais

### 1.1 Stack de Software e Dependências
- **LLM Backbone:** `mistralai/Mistral-7B-Instruct-v0.3`.
- **Quantização:** NF4 (4-bit Normalized Float) via `bitsandbytes`, utilizando double quantization e compute dtype `float16`.
- **Motor de Inferência:** PyTorch 2.4+ com `attn_implementation="sdpa"` (Scaled Dot Product Attention).
- **Armazenamento Vetorial e IR:** ChromaDB com embeddings `nomic-embed-text-v1.5`.
- **Reranking Semântico:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### 1.2 Hardware de Referência e Restrição
- **Ambiente-Alvo:** NVIDIA T4 (16 GB VRAM).
- **Condição de Laboratório (Stress-Test):** Alocação estrita de no máximo **35% da VRAM** (~5.4 GB utilizáveis) para simular dispositivos de borda e forçar severamente condições de Out-of-Memory (OOM).

---

## 2. Inovações Arquiteturais do Cognito Engine

O motor opera através de camadas sobrepostas de interceptação de tensores:

### 2.1 Chunked Prefill e Predição Determinística
Um prefill fatiado (`chunk_size=512`) que previne picos catastróficos de O(N²) ao codificar contextos imensos. Antes de gerar cada token novo, o sistema avalia deterministicamente a pressão da memória física usando a equação da arquitetura de Atenção: `delta_tokens * num_kv_heads * head_dim * num_layers * 2 * 2 bytes`.

### 2.2 RAGAwarePager (Paging por Relevância Semântica)
Em oposição a algoritmos puramente agnósticos focados apenas em Attention Sinks (ex: StreamingLLM), o Cognito endereça segmentos de conhecimento RAG dentro do KV Cache. Se a VRAM vai esgotar, ele aciona o bisturi semântico: fatiando e deletando on-the-fly os tensores que contêm os blocos com o **menor score do Cross-Encoder**.

### 2.3 Curva Temporal de Ebbinghaus (Conversational Memory)
O Cognito incorpora a função logarítmica de decaimento ao peso de sobrevivência do KV Cache em sessões multi-turno. O cálculo $Score \times e^{-0.4 \times lag}$ pune informações atreladas a turnos de conversas antigos, forçando a "Amnésia Contextual Direcionada" para aliviar a carga da memória.

### 2.4 Estabilização Absoluta do RoPE (Rotary Position Embedding)
O avanço mais delicado da pesquisa: quando blocos físicos são apagados do meio do tensor de KV Cache pela paginação semântica, o sistema mantém uma Thread isolada de **tamanho lógico (`logical_len`)**. Os novos *Position IDs* são artificialmente passados ao Mistral corrigidos, impedindo as severas alucinações temporais que afligem implementações customizadas convencionais.

---

## 3. Metodologia Científica e Rigor Acadêmico

### 3.1 Kernel Sandboxing Isolado
Cada fase rodada (`1_ingestao.py`, `3_inferencia.py`) opera em processo isolado pelo `uv run`. Essa blindagem assegura liberação integral dos tensores alocados no CUDA (Pytorch Zombies) e garante que as métricas de pico de VRAM reportadas representem 100% de realidade em hardware limpo.

### 3.2 Semantic Chunking Dinâmico
Quebra sintática inteligente de textos por parágrafos (`\n\n`) e sentenças (`[.!?]`) em vez de *slices* brutos de caracteres. Preserva a integridade e representatividade informacional antes do processamento pelo retriever e gerador.

### 3.3 Testes e Validação Multi-Turno
Benchmarking que agrupa corpus do **TriviaQA** simulando sessões ininterruptas de **Conversational RAG**. Os KPIs monitorados formam a tese empírica de pesquisa: Latência (TTFT e ITL), Volume Paged/Evictado e Métrica Semântica Generativa via **Contains Accuracy**.

---
**Status da Implementação:** Tese Empírica Consolidada.
