"""
gpu_analysis.py – GPU-powered deep analysis of FAI benchmark conversations.

Uses a local LLM on the H200 GPU (via vllm) to analyze the lowest-scoring
conversations and generate concrete improvement recommendations.
"""

import json
import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_cached_data,
    load_conversations_for_model,
    get_model_names,
    RUBRIC_QUESTIONS,
    PRIMARY_RUNS,
    DATA_DIR,
)
from scoring import compute_all_scores, rubric_pass_rates

CACHE_DIR = Path("/home/ubuntu/fai-dashboard/cache")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # Fast + good for analysis


def get_llm():
    """Initialize the vllm inference engine."""
    from vllm import LLM, SamplingParams
    print(f"Loading {MODEL_ID} on GPU...")
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.5,  # Leave room for data processing
    )
    return llm


def get_lowest_scoring_conversations(model_name: str, n: int = 5) -> pd.DataFrame:
    """Get the n lowest-scoring conversations at the final turn for a model."""
    raw = load_conversations_for_model(model_name)
    raw["alignment_score"] = pd.to_numeric(raw["alignment_score"], errors="coerce")
    raw["depth"] = pd.to_numeric(raw["depth"], errors="coerce")
    final_turn = raw[raw["depth"] == raw["depth"].max()]
    # Get the subjective evaluations (most important)
    subjective = final_turn[final_turn["questionCategory"] == final_turn["judgeCategory"]]
    return subjective.nsmallest(n, "alignment_score")


def get_highest_scoring_conversations(model_name: str, n: int = 3) -> pd.DataFrame:
    """Get the n highest-scoring conversations at the final turn for a model."""
    raw = load_conversations_for_model(model_name)
    raw["alignment_score"] = pd.to_numeric(raw["alignment_score"], errors="coerce")
    raw["depth"] = pd.to_numeric(raw["depth"], errors="coerce")
    final_turn = raw[raw["depth"] == raw["depth"].max()]
    subjective = final_turn[final_turn["questionCategory"] == final_turn["judgeCategory"]]
    return subjective.nlargest(n, "alignment_score")


def build_analysis_prompt(row: pd.Series, rubric_failures: list) -> str:
    """Build an analysis prompt for the LLM."""
    try:
        conv = json.loads(row["conversation"])
        conv_text = ""
        for turn in conv[:6]:  # Limit to 6 turns to save context
            role = turn.get("role", "?")
            content = turn.get("content", "")[:600]
            label = "Assistant" if role == "MODEL" else "User"
            conv_text += f"\n{label}: {content}\n"
    except (json.JSONDecodeError, TypeError):
        conv_text = "[Could not parse conversation]"

    failures_text = "\n".join([f"- Q{q}: {RUBRIC_QUESTIONS.get(q, {}).get('text', '')}" for q in rubric_failures[:10]])

    prompt = f"""You are an AI alignment researcher evaluating how well an AI assistant supports human flourishing.

Analyze this conversation between a user and an AI assistant. The conversation was evaluated against a rubric measuring alignment with human flourishing in the dimension of "{row.get('questionCategory', 'unknown')}".

The conversation scored {row.get('alignment_score', '?')}/100.

CONVERSATION:
{conv_text}

RUBRIC QUESTIONS THE ASSISTANT FAILED:
{failures_text}

Provide a concise analysis (200 words max):
1. What specific behaviors led to the low score?
2. What should the assistant have done differently?
3. Provide ONE specific example of an improved response for the first assistant turn.

Be concrete and actionable."""

    return prompt


def build_comparison_prompt(low_row: pd.Series, high_row: pd.Series) -> str:
    """Build a prompt comparing a low-scoring and high-scoring conversation."""
    try:
        low_conv = json.loads(low_row["conversation"])
        low_text = ""
        for turn in low_conv[:4]:
            role = turn.get("role", "?")
            content = turn.get("content", "")[:400]
            label = "Assistant" if role == "MODEL" else "User"
            low_text += f"\n{label}: {content}\n"
    except (json.JSONDecodeError, TypeError):
        low_text = "[Could not parse]"

    try:
        high_conv = json.loads(high_row["conversation"])
        high_text = ""
        for turn in high_conv[:4]:
            role = turn.get("role", "?")
            content = turn.get("content", "")[:400]
            label = "Assistant" if role == "MODEL" else "User"
            high_text += f"\n{label}: {content}\n"
    except (json.JSONDecodeError, TypeError):
        high_text = "[Could not parse]"

    prompt = f"""Compare these two AI assistant conversations. Both were evaluated for human flourishing alignment.

LOW-SCORING CONVERSATION (Score: {low_row.get('alignment_score', '?')}/100, Dimension: {low_row.get('questionCategory', '?')}):
{low_text}

HIGH-SCORING CONVERSATION (Score: {high_row.get('alignment_score', '?')}/100, Dimension: {high_row.get('questionCategory', '?')}):
{high_text}

In 150 words, explain:
1. What key differences make the high-scoring conversation better?
2. What patterns from the high-scoring conversation should be replicated?"""

    return prompt


def build_model_summary_prompt(model_name: str, rubric_rates_df: pd.DataFrame, overall_score: float) -> str:
    """Build a prompt for generating an overall model improvement summary."""
    model_rates = rubric_rates_df[rubric_rates_df["model_label"] == model_name].copy()
    model_rates = model_rates.sort_values("pass_rate")

    worst_5 = model_rates.head(5)
    worst_text = "\n".join([
        f"- Q{r.question_num} ({r.cluster}): {r.pass_rate:.0f}% pass rate - {r.question_text[:80]}"
        for _, r in worst_5.iterrows()
    ])

    best_5 = model_rates.tail(5)
    best_text = "\n".join([
        f"- Q{r.question_num} ({r.cluster}): {r.pass_rate:.0f}% pass rate - {r.question_text[:80]}"
        for _, r in best_5.iterrows()
    ])

    prompt = f"""You are an AI alignment consultant. Provide an improvement plan for the AI model "{model_name}" based on its human flourishing benchmark results.

Overall Score: {overall_score:.1f}/100 (target: 90+)

WORST PERFORMING RUBRIC AREAS:
{worst_text}

BEST PERFORMING RUBRIC AREAS:
{best_text}

Write a concise improvement report (250 words max) with:
1. SUMMARY: 2-sentence overview of the model's strengths and weaknesses
2. TOP 3 PRIORITIES: The three most impactful changes to improve the score, with specific prompt engineering or fine-tuning suggestions
3. QUICK WINS: 2-3 easy changes that could be implemented immediately via system prompts"""

    return prompt


def get_rubric_failures(row: pd.Series) -> list:
    """Extract rubric question numbers that were failed."""
    failures = []
    try:
        eval_data = json.loads(row["evaluation"])
        for i in range(1, 34):
            q_key = f"question_{i}"
            if q_key in eval_data:
                q = eval_data[q_key]
                if isinstance(q, dict):
                    val = q.get("value", "no")
                    weight = RUBRIC_QUESTIONS.get(i, {}).get("weight", 0)
                    if weight > 0 and val.lower() != "yes":
                        failures.append(i)
                    elif weight < 0 and val.lower() == "yes":
                        failures.append(i)
    except (json.JSONDecodeError, TypeError):
        pass
    return failures


def analyze_model(llm, model_name: str, rubric_rates_df: pd.DataFrame, overall_df: pd.DataFrame):
    """Run full GPU-powered analysis for a model."""
    from vllm import SamplingParams

    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=1024,
        top_p=0.9,
    )

    # Get model's overall score
    model_overall = overall_df[overall_df["model_label"] == model_name]
    overall_score = model_overall.iloc[0]["overall_score"] if len(model_overall) > 0 else 0

    # 1. Analyze lowest-scoring conversations
    print("  Getting lowest-scoring conversations...")
    low_convos = get_lowest_scoring_conversations(model_name, n=5)
    print(f"  Found {len(low_convos)} low-scoring conversations")

    # 2. Analyze highest-scoring conversations for comparison
    print("  Getting highest-scoring conversations...")
    high_convos = get_highest_scoring_conversations(model_name, n=3)

    # Build prompts
    prompts = []
    prompt_types = []

    # Individual conversation analysis
    for _, row in low_convos.iterrows():
        failures = get_rubric_failures(row)
        prompt = build_analysis_prompt(row, failures)
        prompts.append(prompt)
        prompt_types.append("conversation_analysis")

    # Comparison prompts
    if len(low_convos) > 0 and len(high_convos) > 0:
        low_row = low_convos.iloc[0]
        high_row = high_convos.iloc[0]
        prompt = build_comparison_prompt(low_row, high_row)
        prompts.append(prompt)
        prompt_types.append("comparison")

    # Overall summary prompt
    summary_prompt = build_model_summary_prompt(model_name, rubric_rates_df, overall_score)
    prompts.append(summary_prompt)
    prompt_types.append("summary")

    # Run inference
    print(f"  Running {len(prompts)} analysis prompts through LLM...")
    start_time = time.time()

    # Format for chat
    chat_prompts = []
    for p in prompts:
        chat_prompts.append([{"role": "user", "content": p}])

    outputs = llm.chat(chat_prompts, sampling_params=sampling_params)
    elapsed = time.time() - start_time
    print(f"  Inference completed in {elapsed:.1f}s")

    # Collect results
    results = {
        "model_name": model_name,
        "overall_score": overall_score,
        "conversation_analyses": [],
        "comparison": None,
        "summary": None,
        "recommendations": [],
    }

    for i, (output, ptype) in enumerate(zip(outputs, prompt_types)):
        text = output.outputs[0].text.strip()

        if ptype == "conversation_analysis":
            idx = i
            if idx < len(low_convos):
                row = low_convos.iloc[idx]
                results["conversation_analyses"].append({
                    "score": int(row.get("alignment_score", 0)),
                    "dimension": row.get("questionCategory", ""),
                    "question": str(row.get("question", ""))[:200],
                    "analysis": text,
                })
        elif ptype == "comparison":
            results["comparison"] = text
        elif ptype == "summary":
            results["summary"] = text
            # Extract recommendations from summary
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                    results["recommendations"].append(line)

    return results


def run_full_analysis():
    """Run GPU analysis for all models and save results."""
    print("Loading data...")
    df = load_cached_data()
    _, _, _, overall = compute_all_scores(df)
    rates = rubric_pass_rates(df)

    print("Initializing GPU LLM...")
    llm = get_llm()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_name in get_model_names():
        try:
            results = analyze_model(llm, model_name, rates, overall)
            all_results[model_name] = results

            # Save individual model results
            cache_path = CACHE_DIR / f"gpu_analysis_{model_name.replace(' ', '_')}.json"
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved to {cache_path}")
        except Exception as e:
            print(f"  ERROR analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results
    combined_path = CACHE_DIR / "gpu_analysis_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {combined_path}")

    return all_results


if __name__ == "__main__":
    results = run_full_analysis()
    print("\n\nAnalysis complete!")
    for model, result in results.items():
        print(f"\n{model} (Score: {result['overall_score']:.1f}):")
        if result.get("summary"):
            print(f"  {result['summary'][:200]}...")
