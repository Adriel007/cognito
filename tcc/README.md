# Cognito: Sistema de Inferência com Paging Dinâmico de KV Cache

**Trabalho de Conclusão de Curso (TCC) — Documentação Técnica de Referência**

O **Cognito** é uma solução de engenharia para Grandes Modelos de Linguagem (LLMs) que implementa gestão de memória de contexto (KV Cache) em nível de aplicação. O sistema é projetado para operar em ambientes com restrição severa de VRAM, transformando a gestão de memória de um gargalo passivo em um mecanismo ativo de filtragem semântica e preservação de integridade lógica.

---

## 1. Especificações Técnicas Centrais

### 1.1 Stack de Software e Dependências
- **LLM Backbone:** `mistralai/Mistral-7B-Instruct-v0.3`.
- **Quantização:** NF4 (4-bit Normalized Float) via `bitsandbytes`, utilizando double quantization e compute dtype `float16`.
- **Motor de Inferência:** PyTorch 2.4+ com `attn_implementation="sdpa"` (Scaled Dot Product Attention).
- **Gerenciamento de Ambiente:** `uv` (PEP 723) para isolamento de interpretadores e dependências transientes.
- **Armazenamento Vetorial:** ChromaDB com embeddings `nomic-embed-text-v1.5`.
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### 1.2 Hardware de Referência
- **GPU:** NVIDIA T4 (16 GB VRAM).
- **Simulação de Restrição:** Alocação restrita a 35% da capacidade física (~5.4 GB utilizáveis) via `set_per_process_memory_fraction`.

---

## 2. Metodologia Científica e Justificativas

### 2.1 Isolamento de Kernel (Kernel Sandboxing)
Diferente da execução linear em notebooks IPython, o Cognito utiliza um protocolo de **Kernel Sandboxing**:
1. Cada fase é persistida via `%%writefile` e executada como um subprocesso isolado via `uv run`.
2. Isso garante a **liberação total de tensores zumbis** e fragmentação de VRAM entre as fases de processamento.
3. Garante que m\u00e9tricas de consumo de memória sejam brutas e não contaminadas por estados anteriores da sessão.

### 2.2 Integridade de Dados (Zero Leakage)
O pipeline implementa uma separação rigorosa de splits do dataset TriviaQA:
- **Corpus (Knowledge Base):** Split `TRAIN`.
- **Avaliação:** Split `VALIDATION`.
- **Justificativa:** Garante que o modelo não recupere informações por "memorização" de contexto idêntico ao de treino durante o retrieval.

---

## 3. Arquitetura do Cognito Engine

O motor de inferência é composto por quatro camadas de gestão de memória que operam de forma orquestrada:

### 3.1 VirtualPageManager (Camada de Persistência)
Implementa o paging básico de KV Cache. Ao atingir o limite físico da VRAM, move blocos de tensores históricos para a RAM do sistema (CPU) de forma não-bloqueante. Reconstrói objetos `DynamicCache` preservando o estado de `seen_tokens` para manter a coerência do RoPE (Rotary Positional Embedding).

### 3.2 AdaptiveVirtualPageManager (Camada Preventiva)
Utiliza uma Média Móvel Exponencial (EMA) para monitorar a pulsação de consumo de VRAM a cada token gerado. Ajusta o limiar de paging dinamicamente (`threshold = ema_vram * (1 + safety_margin)`), reagindo a oscilações no pool de memória do driver CUDA.

### 3.3 PredictiveMemoryPolicy (Camada Preditiva)
Implementa uma política de **Offload Causal**. Estima geometricamente o incremento de VRAM do próximo passo de decodificação antes que ele ocorra.
- **Custo Predictivo:** `delta_tokens * num_kv_heads * head_dim * num_layers * 2 * 2 bytes`.
- **Ação:** O offload é disparado preventivamente se `consumo_atual + estimativa > threshold`.

### 3.4 RAGAwarePager (Camada Semântica)
A inovação central do projeto. Diferente do StreamingLLM (que usa localidade temporal), o **RAGAwarePager** utiliza **Prioridade Semântica**:
- **Segmentação:** Cada passagem do RAG é registrada como um segmento KV endereçável.
- **Evicção por Relevância:** Em caso de pressão de memória, o sistema evicta fisicamente o segmento com menor score de reranking, independentemente de sua posição no prompt.
- **Decaimento de Ebbinghaus:** Para diálogos multi-turno, os scores são ponderados por uma função de esquecimento exponencial para priorizar informações recentes sem descartar fatos críticos.

---

## 4. Otimizações de Performance

- **Automatic Prefix Caching (APC):** O prefixo do sistema (Instruções + Persona) é pré-computado e mantido em "VRAM quente". Todas as queries subsequentes reutilizam este estado, eliminando redundância computacional.
- **Chunked Prefill:** Contextos longos são processados em blocos de 512 tokens para permitir que o Pager intervenha durante a fase de codificação inicial, prevenindo OOMs logo no início da inferência.
- **Noise Pruning:** Metodologia de poda de contextos irrelevantes (ruído) via integração de Reranking Cross-Encoder com evicção por score diretamente no KV Cache.

---

## 5. Estrutura do Pipeline de Execução

1.  **Fase 1 (Ingestão):** Chunking de 1024 caracteres com overlap de 256. Indexação vetorial via ChromaDB.
2.  **Fase 2 (Benchmarking):** Avaliação de base (Zero-shot) em ARC, HellaSwag, MMLU e WinoGrande para validar a integridade do modelo NF4.
3.  **Fase 3 (Inferência Cognito):** Execução do motor RAG com monitoramento ativo de VRAM e coleta de métricas de ablação (OK vs. OOM).

---

## 6. Resultados e Métricas de Sucesso

- **Métricas de Texto:** Exact Match (EM) e F1-Token ( TriviaQA).
- **Métricas de Eficiência:** Capacidade de processar contextos de 40.000 caracteres em budgets de memória de 5.4GB, onde o pipeline baseline sofreria OOM instantâneo.
- **Estabilidade:** Redução drástica de falhas de alocação CUDA via intervenção preditiva do Pager.

---
**Status da Implementação:** Consumado.
