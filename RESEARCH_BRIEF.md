# Flourishing AI Multi-Turn Benchmark: Research Brief

**Prepared for Gloo AI Research**
**February 2026**

---

## Executive Summary

This research brief presents findings from an independent evaluation of the Flourishing AI Multi-Turn (FAI-MT) Benchmark alpha. We analyzed 60,909 evaluation records across 10 frontier AI models, 7 dimensions of human flourishing, and 8 conversation turns. Our analysis covers four areas specified in the product requirements: Model Behavior Analysis, Interpretability and Rubric Evaluation, Bias and Fairness Characterization, and Data Platform Development.

**Key findings:**
- No model reaches the 90-point flourishing threshold; the best performer (Grok 4) scores 73.4 overall.
- Multi-turn scores start dramatically lower than single-turn (41.2 vs 61.1 avg) but exceed single-turn by turn 8 (67.4 avg), largely due to the additive rubric structure.
- 75,873 error cascade instances were detected where early rubric failures persist throughout conversations.
- 56 of 70 model-dimension combinations show statistically significant scoring bias.
- Several rubric weights are misaligned with their actual scoring impact, suggesting weight recalibration is warranted.

---

## 1. Model Behavior Analysis

### 1.1 Overall Performance Rankings

| Rank | Model | Overall Score | First Turn | Last Turn |
|------|-------|--------------|------------|-----------|
| 1 | Grok 4 | 73.4 | 52.7 | 77.7 |
| 2 | GPT OSS 120B | 71.1 | 45.1 | 81.5 |
| 3 | GPT-5 | 66.7 | 42.6 | 72.9 |
| 4 | Claude 3.5 Haiku | 59.5 | 42.4 | 69.7 |
| 5 | DeepSeek-R1 | 58.0 | 37.9 | 67.4 |
| 6 | Claude Sonnet 4 | 56.8 | 36.2 | 63.3 |
| 7 | Claude Opus 4 | 55.1 | 35.6 | 61.9 |
| 8 | Claude 3 Haiku | 53.9 | 43.1 | 61.1 |
| 9 | Llama 4 Scout 109B | 53.1 | 38.6 | 58.7 |
| 10 | GPT-4o | 48.3 | 36.0 | 53.4 |

### 1.2 Single-Turn vs Multi-Turn Comparison

Multi-turn first turns score significantly lower than single-turn subjective scores (avg 41.2 vs 61.1), reflecting the harder, SME-crafted question set in MT. However, by turn 8, multi-turn scores exceed single-turn (avg 67.4 vs 61.1), demonstrating that continued conversation allows models to build and refine responses.

This has important implications for benchmark interpretation: the MT benchmark is measuring a fundamentally different capability (sustained conversational guidance) rather than just a harder version of the ST benchmark.

### 1.3 Error Cascades

We identified 75,873 instances where rubric failures in early turns persist throughout conversations. Key findings:

- **Claude Sonnet 4** shows the highest average cascade severity (10.33), meaning early failures are most likely to persist.
- **GPT-5** shows the lowest cascade severity (9.90), recovering best from early mistakes.
- The most cascading rubric questions relate to **asking for context before prescribing solutions** (Q6, weight=24) and **disclosing assumptions and uncertainties** (Q15, weight=20).
- Error cascades are structurally baked in: once a model fails to ask for context in turn 1, the conversation template provides no natural opportunity to recover.

**Recommendation:** Consider rubric modifications that allow partial credit for late recovery, or weight early-turn failures differently than late-turn failures.

### 1.4 Contextual Drift

Score volatility analysis across conversations reveals:

- **Claude 3 Haiku** is the most stable model (avg volatility 9.74), though it scores lower overall.
- **GPT OSS 120B** shows the highest volatility (17.95) despite ranking 2nd overall, suggesting inconsistent but occasionally excellent performance.
- Only 6.4% of GPT OSS 120B conversations show declining trends, compared to 19.3% for Grok 4.
- Most models show reduced volatility in later turns (turns 6-8), indicating that conversations converge to stable scoring behavior.

**Recommendation:** Report volatility alongside aggregate scores to distinguish consistently good models from inconsistently excellent ones.

---

## 2. Interpretability and Rubric Evaluation

### 2.1 Rubric Question Effectiveness

Analysis of 33 rubric questions revealed widely varying pass rates:

- **Near-universal pass (>95%):** Q1 (safety, 99.3%), Q10 (formatting, 99.9%), Q11 (foster discussion, 96.1%), Q26 (avoid jargon, 98.9%), Q31 (response length matching, 97.1%)
- **Near-universal fail (<5%):** Q5 (align with ALL principles, 0.0%), Q8 (endorse harm, 0.02%), Q14 (refuse without justification, 3.4%)
- **Differentiating questions (20-80% pass):** Q6 (ask for context), Q15 (disclose assumptions), Q17 (cultural sensitivity), Q20 (recommend professionals)

Questions that pass/fail universally provide no differentiating signal. The benchmark's discriminative power comes from roughly 15-20 of the 33 questions.

### 2.2 Rubric Weight Alignment

We computed the point-biserial correlation between each rubric question's pass/fail and the alignment score:

| Question | Assigned Weight | Actual Impact | Correlation | Aligned? |
|----------|----------------|---------------|-------------|----------|
| Q28 (feasible suggestions) | 10 | 38.9 | 0.602 | Underweighted |
| Q15 (disclose assumptions) | 20 | 39.3 | 0.594 | Well-aligned |
| Q11 (foster discussion) | 3 | 38.7 | 0.502 | Severely underweighted |
| Q27 (self-efficacy) | 10 | 38.9 | 0.602 | Underweighted |
| Q1 (refuse harm) | 8 | 49.6 | 0.161 | Low correlation (ceiling) |

**Key finding:** Several low-weight questions (Q11 w=3, Q10 w=3) have high actual impact on scores, suggesting their weights should be increased. Conversely, some high-weight questions like Q1 (w=8) show low correlation because they pass nearly universally, providing no differentiating power.

**Recommendation:** Recalibrate rubric weights based on empirical impact analysis. Consider removing or reworking questions with near-100% or near-0% pass rates.

### 2.3 Rubric Cluster Analysis

The rubric's five semantic clusters contribute differently to score differentiation:

| Cluster | Total Weight | Avg Actual Impact | Status |
|---------|-------------|-------------------|--------|
| Definitional Alignment | 76 (25%) | Moderate | Working as intended |
| Tailoring | 74 (25%) | High | Most discriminative cluster |
| Acknowledgment of Limitations | 70 (25%) | High | Critical differentiator |
| Actionability | 45 (15%) | High | Underweighted relative to impact |
| Structural Quality | 30 (10%) | Moderate | Appropriate weight |

---

## 3. Bias and Fairness

### 3.1 Dimension-Level Bias

56 of 70 model-dimension combinations show statistically significant scoring deviations from the model's overall average (p < 0.05). This is expected given the genuine difficulty differences between dimensions, but the magnitude of bias varies:

- **Consistently disadvantaged dimensions:** Character (avg -6.2 pp below mean), Faith & Spirituality (avg -4.8 pp)
- **Consistently advantaged dimensions:** Physical & Mental Health (avg +8.3 pp above mean), Happiness (avg +5.1 pp)

This pattern is consistent across all models, suggesting the bias reflects genuine difficulty differences in the question set rather than model-specific issues.

### 3.2 Question Difficulty

The 141 unique questions span a wide range of difficulty (avg scores from 15 to 85 across models). Faith & Spirituality questions are the most difficult on average, followed by Character. Questions about Physical & Mental Health are consistently easiest.

### 3.3 Conversation Length Effects

All models benefit significantly from longer conversations, with average score increases of 25-35 points from turn 1 to turn 8. This is largely an artifact of the additive rubric (once points are scored, they are rarely lost). The benchmark as designed inherently advantages longer conversations.

**Recommendation:** Consider whether the additive rubric structure accurately reflects conversational quality, or whether a per-turn independent scoring approach would better capture quality at each stage.

---

## 4. Data Platform

### 4.1 Platform Built

A comprehensive research data platform was implemented with:

- **Data Lake:** Parquet-cached data integrating all 10 model evaluation runs (60,909 rows, 114 features)
- **Scoring Engine:** Full replication of the FAI scoring methodology (conversation, dimension, categorical, and turn-level aggregation)
- **Interactive Dashboard:** 12-page Streamlit application with model comparison, dimension analysis, turn-by-turn tracking, rubric analysis, conversation browsing, improvement recommendations, ST/MT comparison, error cascades, contextual drift, bias analysis, weight validation, and data provenance
- **GPU Analysis Pipeline:** vllm-powered local LLM analysis of lowest-scoring conversations on NVIDIA H200

### 4.2 Data Quality

No missing values were detected in core evaluation columns. Data volume varies by model due to different run configurations (from 439 rows for smaller Haiku runs to 13,456 rows for GPT-4o). The depth distribution is uniform across turns 1-8.

---

## 5. ML/NLP Root Cause Analysis & Solutions

### 5.1 Critical: First-Turn Cold Start (est. +15-25 points)
**Root Cause:** RLHF helpfulness bias (Ouyang et al., 2022) trains models to answer immediately rather than gather context. Instruction-tuning data consists of single-turn (prompt, response) pairs that don't model context-gathering behavior.
**Solution:** System prompt engineering ("ALWAYS begin by asking 1-2 clarifying questions"), DPO fine-tuning on context-gathering conversation pairs (Rafailov et al., 2023), RAG-powered dimension-specific context templates.

### 5.2 Critical: Transparency Deficit (est. +8-12 points)
**Root Cause:** RLHF penalizes hedging — annotators rate confident responses higher than uncertain ones (Kadavath et al., 2022). Models lack training to verbalize internal uncertainty. No explicit instruction to disclose AI limitations.
**Solution:** Structured output templates with "Assumptions" and "Limitations" sections. Uncertainty-aware decoding that detects high token-level entropy and appends hedges (Lin et al., 2023).

### 5.3 Critical: Faith & Spirituality Gap (est. +10-20 points)
**Root Cause:** RLHF neutrality bias — disagreement among annotators on faith content leads models to optimize for "least objectionable" secular responses (Sorensen et al., 2024). Training data skews toward informational religious content over applied theological reasoning. Safety guardrails over-correct by flagging religious content as controversial.
**Solution:** Values-aligned fine-tuning on pastoral counseling transcripts and theological ethics cases. Constitutional AI with faith-specific principles (Bai et al., 2022). Dimension-specific system prompts that engage substantively with spiritual perspectives.

### 5.4 High: Error Cascades (est. +5-8 points)
**Root Cause:** Autoregressive path dependency — transformer induction heads (Olsson et al., 2022) copy and extend established conversation patterns. Once a "confident advisor" pattern is set in turn 1, subsequent turns follow suit.
**Solution:** Turn-aware system prompts with per-turn behavior checklists. Mid-conversation prompt injection reminding the model of unchecked rubric behaviors. Rubric modification to allow partial credit for late recovery.

### 5.5 High: Citation Gap (est. +6-10 points)
**Root Cause:** Training data uses vague attribution ("studies show..."). Post-training safety work discourages specific citations due to hallucination risk (Ji et al., 2023). No retrieval mechanism in base LLMs.
**Solution:** RAG integration with curated per-dimension citation databases. System prompt with pre-loaded verified source lists. DPO training on citation-rich conversation pairs with post-generation verification against a source whitelist.

### 5.6 High: Professional Referral Gap (est. +6-10 points)
**Root Cause:** RLHF "self-sufficiency" bias — annotators rate referrals as "less helpful." Models are optimized to provide complete answers rather than acknowledge limits.
**Solution:** System prompt with professional referral rules (dimension → professional type mapping). RAG-powered referral templates with specific hotlines and resource finders.

### 5.7 Implementation Roadmap

| Phase | Timeframe | Actions | Est. Score Impact |
|-------|-----------|---------|-------------------|
| Phase 1: System Prompts | 1-2 weeks | Context-gathering, transparency, referral, citation instructions | +15-25 pts |
| Phase 2: RAG Integration | 2-4 weeks | Per-dimension citation DBs, referral templates, resource libraries | +6-10 pts |
| Phase 3: Fine-Tuning | 4-8 weeks | DPO on context-gathering, citation, faith-aligned conversations | +10-20 pts |
| Phase 4: Benchmark Fixes | 2-4 weeks | IRT weight recalibration, per-turn scoring, expand Faith questions | Better measurement |

**Total estimated improvement: +30-55 points**, potentially pushing top models past the 90-point threshold.

---

## 6. Benchmark Improvement Recommendations

### Priority 1: Rubric Recalibration
- Remove or rework questions with near-100% or near-0% pass rates (Q5, Q8, Q10, Q14, Q30)
- Increase weights for high-impact, low-weight questions (Q11, Q28, Q27)
- Apply Item Response Theory (IRT) for data-driven weight calibration
- Consider dimension-specific rubric variations as recommended by SMEs

### Priority 2: Scoring Methodology
- The additive rubric creates inherent conversation-length bias
- Consider per-turn independent scoring alongside cumulative scoring
- Track behavior-onset timing (when each rubric behavior first appears)
- Report score volatility and trend direction alongside aggregate scores

### Priority 3: Fairness Improvements
- Ensure question difficulty is balanced across dimensions
- Expand Faith & Spirituality question count (currently only 9 questions, 5.6% of dataset)
- Add persona diversity within each dimension

### Priority 5: Interpretability
- Publish rubric question pass rates alongside scores for transparency
- Include confidence intervals on reported scores
- Document which rubric questions differentiate models vs. which are universally passed/failed

---

## Appendix: Methodology

- **Scoring:** Geometric mean aggregation across 7 dimensions; arithmetic mean within dimensions; weighted rubric scoring per conversation
- **Error Cascades:** Tracked rubric question pass/fail across turns; cascade defined as failure at turn 1 persisting through final turn
- **Contextual Drift:** Linear regression of alignment scores across turns; standard deviation as volatility measure
- **Bias Analysis:** One-sample t-tests comparing dimension scores to model means; p < 0.05 significance threshold
- **Weight Validation:** Point-biserial correlation between rubric pass/fail and alignment scores; comparison of assigned weights to empirical impact
- **GPU Analysis:** Qwen2.5-7B-Instruct via vllm on NVIDIA H200; analyzed 5 lowest-scoring conversations per model
