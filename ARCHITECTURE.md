# FAI Benchmark Dashboard – Architecture & Technical Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │ 39 CSV Files      │  │ White Paper / PRD                │ │
│  │ (~1.5 GB)         │  │ (Published ST scores, rubric)    │ │
│  │ 10 model runs     │  │                                  │ │
│  └────────┬─────────┘  └──────────────┬───────────────────┘ │
└───────────┼────────────────────────────┼────────────────────┘
            │                            │
            ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  data_loader.py                                       │   │
│  │  - CSV ingestion with PRIMARY_RUNS mapping            │   │
│  │  - JSON parsing: evaluation → 33 rubric columns       │   │
│  │  - JSON parsing: conversation → turn data             │   │
│  │  - Rubric question metadata (RUBRIC_QUESTIONS dict)   │   │
│  │  - MD5-based Parquet cache for fast reload             │   │
│  │  - Model/dimension helper functions                    │   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  cache/fai_data_*.parquet                             │   │
│  │  - 60,909 rows × 114 columns                         │   │
│  │  - Excludes conversation/evaluation text (size opt)   │   │
│  │  - Auto-invalidated on source file changes            │   │
│  └───────────────────────┬──────────────────────────────┘   │
└──────────────────────────┼──────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ scoring.py   │ │ analysis.py  │ │gpu_analysis.py│
│              │ │              │ │              │
│ FAI Scoring  │ │ PRD Analyses │ │ LLM-Powered  │
│ Methodology  │ │              │ │ Analysis     │
│              │ │ - ST vs MT   │ │              │
│ - Conv Score │ │ - Cascades   │ │ - vllm +     │
│ - Dim Score  │ │ - Drift      │ │   Qwen2.5-7B │
│ - Cat Score  │ │ - Bias       │ │ - Conversation│
│ - Turn Score │ │ - Weights    │ │   analysis   │
│ - Overall    │ │ - Provenance │ │ - NVIDIA H200│
│              │ │              │ │   GPU        │
│ - Rubric     │ │              │ │              │
│   pass rates │ │              │ │ Results →    │
│ - Dim matrix │ │              │ │ cache/*.json │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  app.py (Streamlit Dashboard)                         │   │
│  │                                                       │   │
│  │  12 Pages:                                            │   │
│  │  Sprint 0 (Core Dashboard):                           │   │
│  │    1. Model Leaderboard                               │   │
│  │    2. Dimension Deep Dive                             │   │
│  │    3. Turn-by-Turn Analysis                           │   │
│  │    4. Rubric Failure Analysis                         │   │
│  │    5. Conversation Explorer                           │   │
│  │    6. Improvement Recommendations                     │   │
│  │                                                       │   │
│  │  Sprint 1 (PRD Requirements):                         │   │
│  │    7. ST vs MT Comparison                             │   │
│  │    8. Error Cascade Detection                         │   │
│  │    9. Contextual Drift Analysis                       │   │
│  │   10. Bias & Fairness                                 │   │
│  │   11. Rubric Weight Validation                        │   │
│  │   12. Data Provenance                                 │   │
│  │                                                       │   │
│  │  Caching: @st.cache_data(ttl=3600)                    │   │
│  │  Visualization: Plotly (interactive charts)           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
CSV Files → data_loader.py → Parquet Cache
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              scoring.py    analysis.py   gpu_analysis.py
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                              app.py
                                  │
                                  ▼
                        Streamlit Dashboard
                         (localhost:8501)
```

## Key Design Decisions

### 1. Parquet Caching Strategy
**Decision:** Cache parsed data as Parquet, excluding heavy text columns (conversation, evaluation).
**Rationale:** The 10 primary CSV files total ~1.2 GB. Parsing JSON in 60,909 rows takes ~15 seconds. The Parquet cache reduces reload to <2 seconds. Text columns are loaded on-demand (Conversation Explorer page only) via `load_conversations_for_model()`.
**Cache invalidation:** MD5 hash of file sizes and modification times. Auto-rebuilds when source files change.

### 2. Scoring Methodology
**Decision:** Replicate the exact FAI scoring methodology from the white paper.
**Rationale:** Ensures our computed scores can be validated against the `alignment_score` column in the data. The scoring hierarchy (Conversation → Dimension → Categorical → Turn → Overall) uses geometric mean at aggregation boundaries to penalize imbalanced performance, as specified in the white paper.

### 3. Analysis Module Separation
**Decision:** Separate `analysis.py` from `scoring.py` for PRD-specific analyses.
**Rationale:** `scoring.py` replicates the published methodology and should remain stable. `analysis.py` contains novel analyses (cascades, drift, bias, weight validation) that are research outputs and may evolve.

### 4. GPU Analysis Architecture
**Decision:** Use vllm with Qwen2.5-7B-Instruct for conversation analysis, running offline and caching results as JSON.
**Rationale:** The H200 GPU (144 GB VRAM) can run much larger models, but 7B provides sufficient quality for analytical tasks while allowing fast batch processing of 70+ prompts across 10 models. Results are cached to avoid re-running expensive inference during dashboard use.

### 5. Streamlit over Dash/React
**Decision:** Streamlit for the dashboard framework.
**Rationale:** Streamlit provides rapid development for data-science dashboards with built-in caching, reactive widgets, and Plotly integration. The research-oriented audience benefits from the notebook-like interaction model. Trade-off: less customizable than React, but development time is 3-5x faster.

## File Structure

```
fai-dashboard/
├── app.py                  # Main Streamlit dashboard (12 pages)
├── data_loader.py          # Data ingestion, JSON parsing, caching
├── scoring.py              # FAI scoring methodology replication
├── analysis.py             # PRD-required analyses (cascades, drift, bias, etc.)
├── gpu_analysis.py         # GPU-powered LLM conversation analysis
├── requirements.txt        # Python dependencies
├── README.md               # User-facing documentation
├── RESEARCH_BRIEF.md       # Research findings summary
├── ARCHITECTURE.md         # This file
├── cache/
│   ├── fai_data_*.parquet  # Cached evaluation data (~50 MB)
│   ├── gpu_analysis_*.json # Per-model GPU analysis results
│   └── gpu_analysis_all.json
└── venv/                   # Python 3.12 virtual environment
```

## Module API Reference

### data_loader.py
| Function | Returns | Description |
|----------|---------|-------------|
| `load_raw_data(force_reload)` | DataFrame (60,909 × 116) | Full data with parsed eval columns |
| `load_cached_data()` | DataFrame (60,909 × 114) | Cached data without text columns |
| `load_conversations_for_model(name)` | DataFrame | Raw CSV with conversation text |
| `get_rubric_info()` | dict | Rubric question metadata |
| `get_model_names()` | list[str] | 10 model names |
| `get_dimension_names()` | list[str] | 7 dimension names |

### scoring.py
| Function | Returns | Description |
|----------|---------|-------------|
| `compute_all_scores(df)` | tuple(4 DataFrames) | Full scoring pipeline |
| `rubric_pass_rates(df)` | DataFrame | Pass rate per (model, question) |
| `rubric_pass_rates_by_dimension(df)` | DataFrame | Pass rate per (model, question, dimension) |
| `dimension_score_matrix(dim_scores, depth)` | DataFrame | Model × dimension matrix |

### analysis.py
| Function | Returns | Description |
|----------|---------|-------------|
| `st_vs_mt_comparison(turn_scores)` | DataFrame | ST vs MT score comparison |
| `detect_error_cascades(df)` | DataFrame | Rubric failures persisting across turns |
| `compute_score_volatility(df)` | DataFrame | Per-conversation score stability metrics |
| `persona_bias_analysis(df)` | DataFrame | Dimension-level statistical bias tests |
| `conversation_length_fairness(df)` | DataFrame | Score by conversation depth |
| `question_difficulty_analysis(df)` | DataFrame | Per-question difficulty metrics |
| `rubric_weight_sensitivity(df)` | DataFrame | Weight vs actual impact analysis |
| `data_provenance_summary(df)` | dict | Data lineage and quality summary |

### gpu_analysis.py
| Function | Description |
|----------|-------------|
| `run_full_analysis()` | Runs LLM analysis for all 10 models |
| `analyze_model(llm, model, rates, overall)` | Per-model conversation analysis |

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.12 |
| Dashboard | Streamlit | 1.54 |
| Visualization | Plotly | 6.5 |
| Data Processing | Pandas | 2.3 |
| Storage | Apache Parquet | via PyArrow 23.0 |
| Statistics | SciPy | 1.17 |
| GPU Inference | vllm | 0.15 |
| LLM Model | Qwen2.5-7B-Instruct | — |
| GPU | NVIDIA H200 | 144 GB VRAM |

## Running the System

```bash
# Start the dashboard
cd /home/ubuntu/fai-dashboard
source venv/bin/activate
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Re-run GPU analysis (optional, results cached)
python gpu_analysis.py

# Rebuild data cache (if source CSVs change)
python data_loader.py
```
