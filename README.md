# Flourishing AI Benchmark Dashboard

Interactive analysis dashboard for Gloo's Flourishing AI (FAI) multi-turn benchmark data. Evaluates how well frontier AI models promote human flourishing across 7 dimensions.

## Quick Start

```bash
cd /home/ubuntu/fai-dashboard

# Activate the virtual environment
source venv/bin/activate

# Run the dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

Then open **http://localhost:8501** in your browser.

## Dashboard Pages

### 1. Model Leaderboard
- Overall rankings of 10 frontier AI models
- Heatmap of model x dimension scores at the final turn
- Grouped bar chart comparison across all 7 dimensions

### 2. Dimension Deep Dive
- Select any of the 7 flourishing dimensions
- Subjective vs tangential score breakdown per model
- Score progression over 8 conversation turns
- Hardest rubric questions for each dimension (heatmap)

### 3. Turn-by-Turn Analysis
- Line charts showing score evolution across 8 conversation turns
- Subjective vs tangential breakdown per model
- Turn 1 to Turn 2 jump analysis (all models nearly double their score)
- Per-model dimension-by-turn heatmap

### 4. Rubric Failure Analysis
- Full 33-question rubric pass/fail heatmap across all models
- Key differentiators: what separates top-3 from bottom-3 models
- Pass rates by rubric cluster (Safety, Definitional Alignment, Actionability, Tailoring, Structural Quality)

### 5. Conversation Explorer
- Browse individual multi-turn conversations
- Filter by model, dimension, and score range
- Side-by-side: full conversation text + rubric evaluation with pass/fail icons
- Pagination for large result sets

### 6. Improvement Recommendations
- Per-model priority improvement list (highest-impact rubric failures)
- Detailed actionable recommendations for each rubric question
- Gap analysis comparing selected model vs the top performer
- GPU-powered deep analysis: LLM-generated conversation-level insights and improvement summaries

## Data

The dashboard loads benchmark data from:
```
/home/ubuntu/FAI MT Baylor Shared Folder/FAI MT Baylor Shared Folder/
```

**10 Models Evaluated:**
- Claude 3 Haiku, Claude 3.5 Haiku, Claude Opus 4, Claude Sonnet 4
- DeepSeek-R1, GPT OSS 120B, GPT-4o, GPT-5
- Grok 4, Llama 4 Scout 109B

**7 Dimensions of Human Flourishing:**
- Character, Close Relationships, Faith & Spirituality
- Finances, Happiness, Meaning & Purpose, Physical & Mental Health

**60,909 evaluation rows** across 8 conversation turns per question.

## Scoring Methodology

Implements the FAI scoring methodology from the white paper:
1. **Conversation Score**: Weighted rubric pass/fail (33 questions, weights from -100 to +24)
2. **Dimension Score**: Arithmetic mean per (model, dimension, depth, subjective/tangential)
3. **Categorical Score**: Geometric mean across 7 dimensions
4. **Turn Score**: Geometric mean of subjective and tangential categorical scores
5. **Overall Score**: Weighted average of turn scores (configurable gamma discount)

## GPU Analysis

The `gpu_analysis.py` script uses an NVIDIA H200 GPU with vllm to run Qwen2.5-7B-Instruct for deep conversation analysis:
- Analyzes lowest-scoring conversations for each model
- Compares low vs high scoring conversations
- Generates model-specific improvement reports
- Results are cached in `/home/ubuntu/fai-dashboard/cache/`

To re-run GPU analysis:
```bash
source venv/bin/activate
python gpu_analysis.py
```

## Project Structure

```
fai-dashboard/
  app.py              # Streamlit dashboard (main entry point)
  data_loader.py      # Data ingestion, JSON parsing, Parquet caching
  scoring.py          # FAI scoring methodology implementation
  gpu_analysis.py     # GPU-powered LLM conversation analysis
  requirements.txt    # Python dependencies
  cache/              # Parquet cache + GPU analysis results
  venv/               # Python virtual environment
```

## References

- White Paper: "Flourishing AI: Extending to Multi Turn Settings" (Stykland et al., 2025)
- Benchmark Methodology Summary (Valkyrie Intelligence, April 2025)
- Baylor Kickoff Presentation (January 27, 2026)
