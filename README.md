# Graph RAG vs. PageRank: An Empirical Evaluation Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Cytoscape.js](https://img.shields.io/badge/Cytoscape.js-3.28-ea580c.svg)](https://js.cytoscape.org/)
[![Pytest](https://img.shields.io/badge/Unit%20Tests-8%20Passed-emerald.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-slate.svg)](LICENSE)

An interactive, multi-paradigm Information Retrieval (IR) research platform developed for a **Bachelor Thesis in Computer Science**. 

The platform bridges foundational graph-theoretic ranking (**Classical PageRank**, Brin & Page 1998) with modern neural retrieval (**Dense Semantic RAG**, Lewis et al. 2020) and state-of-the-art rank fusion (**Reciprocal Rank Fusion - RRF**, Cormack et al. 2009). It features real-time topology manipulation, deterministic Large Language Model (LLM) answer synthesis, and an automated experimental benchmarking suite across multiple academic domains.

---

## 📑 Table of Contents
- [Theoretical Overview](#-theoretical-overview)
- [Key Features](#-key-features)
- [Retrieval Paradigms Supported](#-retrieval-paradigms-supported)
- [Experimental Benchmarks](#-experimental-benchmarks)
- [System Architecture](#-system-architecture)
- [Interactive Topology Sandbox](#-interactive-topology-sandbox)
- [Installation & Quickstart](#-installation--quickstart)
- [Verification & Unit Testing](#-verification--unit-testing)
- [Citation](#-citation)

---

## 🔬 Theoretical Overview

Web search historically prioritized **global link authority** modeled as a stationary distribution over a Markov chain (PageRank). Conversely, contemporary Generative AI systems prioritize **local semantic relevance** via metric embedding spaces (RAG). 

This research platform investigates the trade-offs, vulnerabilities, and synergies between both paradigms:

1. **The Incommensurate Scores Problem:** Graph centrality probabilities ($\pi_i \in (0, 1)$), lexical TF-IDF scores, and dense cosine similarities occupy disparate distributions. We evaluate Linear Score Fusion vs. Rank-level aggregation.
2. **Adversarial Robustness (Link Manipulation):** Simulating topological spam (Link Farms) against topological blindness in dense vector spaces.
3. **Hallucination Mitigation in RAG:** Utilizing strict deterministic decoding ($T = 0.0$) and pre-retrieval semantic thresholding ($\text{sim} \ge 0.30$) to prevent out-of-domain knowledge corruption.

---

## ⚡ Key Features

* **Multi-Domain Scientific Datasets:** Modular JSON-driven citation networks spanning **NLP & LLMs**, **Medical AI**, and **Astrophysics & Cosmology**.
* **Interactive Cytoscape.js Visualizer:** "Details-on-Demand" UI with dynamic node scaling, hover title tooltips, and click-based multi-metric score breakdown.
* **Live Topology Sandbox:** Real-time in-canvas citation edge insertion (double-click to connect) and deletion, immediately recalculating the underlying transition probability matrix.
* **Deterministic LLM Generation:** Zero-temperature ($T = 0.0$) grounded academic synthesis powered by Meta Llama 3 via high-throughput Groq LPUs.
* **Automated Evaluation Suite:** Ground-truth benchmarking producing **Precision@1 (P@1)**, **Mean Reciprocal Rank (MRR)**, **NDCG@3**, and sub-20ms latency metrics.

---

## 📐 Retrieval Paradigms Supported

| Paradigm | Mathematical Formulation | Core Principle |
| :--- | :--- | :--- |
| **1. Classical PageRank** | $\text{Score} = \text{TF-IDF}(q, d) \times \boldsymbol{\pi}(d)$ | Modulates stationary Markov authority with query term frequency (Brin & Page, 1998). |
| **2. Dense Semantic RAG** | $\text{Score} = \cos(\mathbf{e}_q, \mathbf{e}_d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|\|\mathbf{e}_d\|}$ | Bi-encoder metric similarity using `all-MiniLM-L6-v2` over concatenated Titles and Abstracts. |
| **3. Linear Hybrid Fusion** | $\text{Score} = \alpha \cdot \text{Score}_{\text{PR}} + (1 - \alpha) \cdot \text{Score}_{\text{RAG}}$ | Parametric score-level combination controlled via dynamic UI slider ($\alpha \in [0, 1]$). |
| **4. Reciprocal Rank Fusion (RRF)** | $\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}, \quad k = 60$ | Non-parametric rank-level aggregation (Cormack et al., SIGIR 2009). Robust to scale distortions. |

---

## 📊 Experimental Benchmarks

Automated quantitative evaluation over domain ground-truth queries (reproduced via `python -m scripts.run_experiments`):

| Domain | Retrieval Paradigm | P@1 | MRR | NDCG@3 | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NLP & LLMs** | Classical PageRank | 50.0% | 0.708 | 0.783 | 201.8 |
| | Dense Semantic RAG | **100.0%** | **1.000** | **1.000** | 15.0 |
| | Linear Hybrid Fusion | **100.0%** | **1.000** | **1.000** | 15.4 |
| | Hybrid (RRF) | **100.0%** | **1.000** | **1.000** | 15.3 |
| **Medical AI** | Classical PageRank | 25.0% | 0.562 | 0.565 | 40.5 |
| | Dense Semantic RAG | **100.0%** | **1.000** | **1.000** | 14.5 |
| | Linear Hybrid Fusion | 75.0% | 0.875 | 0.908 | 14.7 |
| | Hybrid (RRF) | 75.0% | 0.875 | 0.908 | 17.3 |
| **Astrophysics** | Classical PageRank | 50.0% | 0.750 | 0.815 | 36.7 |
| | Dense Semantic RAG | **100.0%** | **1.000** | **1.000** | 16.8 |
| | Linear Hybrid Fusion | 75.0% | 0.875 | 0.908 | 17.8 |
| | Hybrid (RRF) | 75.0% | 0.875 | 0.908 | 14.7 |

---

## 🏗️ System Architecture

```text
PageRank-Web-App/
├── data/                               # Domain citation networks & benchmark datasets
│   ├── dataset_nlp.json                # Foundational NLP & Transformer papers
│   ├── dataset_medical.json            # Clinical informatics & bio-NLP papers
│   └── dataset_astrophysics.json       # Gravitational radiation & cosmology papers
├── src/
│   ├── models.py                       # Pydantic data contracts (API & Graph models)
│   ├── data_loader.py                  # In-memory JSON ingestion & dataset management
│   ├── algorithms/
│   │   ├── pagerank.py                 # Power Iteration, stochastic adjustment & TF-IDF
│   │   ├── rag.py                      # Dense embeddings with in-memory caching
│   │   ├── hybrid.py                   # Linear score fusion & RRF rank aggregation
│   │   ├── benchmark.py                # Formal IR metric evaluation (MRR, NDCG@3, P@1)
│   │   └── llm.py                      # Groq LPU client (T=0.0 strict grounding)
│   └── api/
│       └── routes.py                   # REST endpoints (/search, /datasets, /benchmark)
├── static/
│   ├── index.html                      # Semantic layout & multi-paradigm controls
│   ├── css/style.css                   # Responsive layout & Cytoscape styles
│   └── js/app.js                       # Client controller, canvas events & dynamic mapping
├── scripts/
│   └── run_experiments.py             # CLI runner exporting results to JSON & LaTeX tables
├── tests/
│   └── test_algorithms.py             # Pytest suite verifying Markov & IR properties
├── main.py                             # Application entrypoint & static file server
├── requirements.txt                    # Python dependencies
└── pytest.ini                          # Testing configuration
```

---

## 🎮 Interactive Topology Sandbox

The Cytoscape.js canvas functions as an active experiment sandbox:
* **Inspect Paper:** Click any node to review its Title, Abstract, Year, and detailed **Score Breakdown** (PageRank, RAG, and RRF normalized and raw metrics).
* **Create Citation:** **Double-click** Node $A$ (highlighted in purple), then **single-click** Node $B$ to dynamically add a directed edge ($A \to B$).
* **Delete Citation:** **Click** any edge line to delete it immediately.
* **Reset Topology:** Click **`↺ Reset Benchmark`** to revert the citation network to its peer-reviewed baseline.

---

## 🚀 Installation & Quickstart

### Prerequisites
* Python 3.10 or higher
* A free [Groq Cloud API Key](https://console.groq.com/)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/tolis-snr/PageRank-Web-App.git
cd PageRank-Web-App

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 3. Launch Application
```bash
uvicorn main:app --reload
```
Navigate to **`http://127.0.0.1:8000`** in your browser. API interactive Swagger documentation is available at **`http://127.0.0.1:8000/docs`**.

---

## 🧪 Verification & Unit Testing

The algebraic and information retrieval properties of the algorithmic core are verified via **Pytest**:

```bash
pytest -v
```

### Verified Properties (8/8 Passed):
* **Perron-Frobenius Probability Conservation:** $\sum \pi_i = 1.0$ and $\pi_i > 0$ strictly hold.
* **Topological Convergence:** Dead-ends and directed cycles resolve stably without division errors.
* **Lexical Hard Filtering:** Non-matching documents strictly evaluate to $0.0$ regardless of link authority.
* **RRF Permutation Symmetry & Monotonicity:** Verifies $RRF_{(1, 2)} \equiv RRF_{(2, 1)}$ and strictly preserved rank ordering.
* **IR Evaluation Calculations:** Mathematical correctness of P@1, MRR, and NDCG@3 implementations.

---

## 📜 Citation

If you use this evaluation platform or codebase in your academic research, please cite:

```bibtex
@bachelorthesis{snr2026graphrag,
  author       = {Your Full Name},
  title        = {Graph RAG vs. PageRank: The Evolution of Web Information Retrieval from Link Authority to Semantic Relevance},
  school       = {Department of Computer Science, University Name},
  year         = {2026},
  month        = {November},
  type         = {Bachelor's Thesis}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
