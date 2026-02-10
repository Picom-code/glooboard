"""
analysis.py – Advanced analysis modules for PRD requirements.

Covers:
  1. Single-Turn vs Multi-Turn comparison
  2. Error cascade detection
  3. Contextual drift analysis
  4. Bias & fairness assessment
  5. Rubric weight validation
  6. Data provenance
"""

import numpy as np
import pandas as pd
from scipy import stats
from data_loader import RUBRIC_QUESTIONS, get_dimension_names, get_model_names


# ======================================================================
# 1. SINGLE-TURN vs MULTI-TURN COMPARISON
# ======================================================================

# Published single-turn scores from the white paper (Table 5)
SINGLE_TURN_SCORES = {
    "Claude 3.5 Haiku":     {"overall": 51.39, "subjective": 48.73},
    "Claude Opus 4":        {"overall": 65.10, "subjective": 57.12},
    "Claude Sonnet 4":      {"overall": 63.42, "subjective": 57.41},
    "GPT OSS 120B":         {"overall": 70.39, "subjective": 69.37},
    "GPT-4o":               {"overall": 57.66, "subjective": 55.06},
    "GPT-5":                {"overall": 72.55, "subjective": 67.52},
    "Grok 4":               {"overall": 75.62, "subjective": 74.97},
    "Llama 4 Scout 109B":   {"overall": 54.50, "subjective": 58.43},
}


def st_vs_mt_comparison(turn_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare single-turn published scores with multi-turn first/last turn scores."""
    records = []
    for model, st_data in SINGLE_TURN_SCORES.items():
        model_turns = turn_scores[turn_scores["model_label"] == model]
        if model_turns.empty:
            continue

        first_turn = model_turns[model_turns["depth"] == 1]
        last_turn = model_turns[model_turns["depth"] == model_turns["depth"].max()]

        mt_first = first_turn["turn_score"].values[0] if len(first_turn) > 0 else None
        mt_last = last_turn["turn_score"].values[0] if len(last_turn) > 0 else None

        records.append({
            "model": model,
            "st_overall": st_data["overall"],
            "st_subjective": st_data["subjective"],
            "mt_first_turn": round(mt_first, 2) if mt_first else None,
            "mt_last_turn": round(mt_last, 2) if mt_last else None,
            "st_to_mt_first_delta": round(mt_first - st_data["subjective"], 2) if mt_first else None,
            "st_to_mt_last_delta": round(mt_last - st_data["subjective"], 2) if mt_last else None,
            "mt_improvement": round(mt_last - mt_first, 2) if mt_first and mt_last else None,
        })
    return pd.DataFrame(records)


# ======================================================================
# 2. ERROR CASCADE DETECTION
# ======================================================================

def detect_error_cascades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect error cascades: conversations where early rubric failures
    persist or worsen across turns.

    Groups by (model, question, judgeCategory) and tracks rubric pass/fail
    across depths to find cascading patterns.
    """
    # We need to identify unique conversations
    # A conversation is identified by (model_label, question, judgeCategory)
    conv_cols = ["model_label", "question", "judgeCategory"]

    rubric_cols = [f"q{i}_pass" for i in range(1, 34) if f"q{i}_pass" in df.columns]

    records = []
    for keys, group in df.groupby(conv_cols):
        model, question, judge_cat = keys
        group = group.sort_values("depth")
        if len(group) < 3:
            continue

        for col in rubric_cols:
            qnum = int(col.replace("q", "").replace("_pass", ""))
            weight = RUBRIC_QUESTIONS.get(qnum, {}).get("weight", 0)
            if weight <= 0:
                continue  # Only track positive-weight questions

            values = group[col].values
            depths = group["depth"].values

            # Cascade: failed at turn 1 AND still failed at last turn
            if len(values) >= 2 and not values[0] and not values[-1]:
                # Count how many turns it stayed failed
                consecutive_fails = 0
                for v in values:
                    if not v:
                        consecutive_fails += 1
                    else:
                        break

                records.append({
                    "model_label": model,
                    "question": question[:100],
                    "judgeCategory": judge_cat,
                    "rubric_q": qnum,
                    "rubric_text": RUBRIC_QUESTIONS.get(qnum, {}).get("text", "")[:80],
                    "weight": weight,
                    "first_turn_pass": bool(values[0]),
                    "last_turn_pass": bool(values[-1]),
                    "consecutive_early_fails": consecutive_fails,
                    "total_fails": int((~pd.Series(values)).sum()),
                    "total_turns": len(values),
                    "cascade_severity": round(int((~pd.Series(values)).sum()) / len(values) * weight, 2),
                })

    result = pd.DataFrame(records)
    return result


def cascade_summary_by_model(cascade_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize cascade patterns per model."""
    if cascade_df.empty:
        return pd.DataFrame()
    summary = cascade_df.groupby("model_label").agg(
        total_cascades=("cascade_severity", "count"),
        avg_severity=("cascade_severity", "mean"),
        avg_consecutive_fails=("consecutive_early_fails", "mean"),
        most_common_q=("rubric_q", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None),
    ).reset_index().sort_values("avg_severity", ascending=False)
    return summary


def cascade_by_rubric_question(cascade_df: pd.DataFrame) -> pd.DataFrame:
    """Which rubric questions cascade most often?"""
    if cascade_df.empty:
        return pd.DataFrame()
    return cascade_df.groupby(["rubric_q", "rubric_text", "weight"]).agg(
        cascade_count=("cascade_severity", "count"),
        avg_severity=("cascade_severity", "mean"),
        models_affected=("model_label", "nunique"),
    ).reset_index().sort_values("cascade_count", ascending=False)


# ======================================================================
# 3. CONTEXTUAL DRIFT ANALYSIS
# ======================================================================

def compute_score_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure score volatility across turns for each conversation.
    High volatility = unstable/drifting behavior.
    """
    conv_cols = ["model_label", "question", "judgeCategory"]
    records = []

    for keys, group in df.groupby(conv_cols):
        model, question, judge_cat = keys
        group = group.sort_values("depth")
        scores = group["alignment_score"].values

        if len(scores) < 3:
            continue

        # Score trajectory metrics
        score_std = np.std(scores)
        score_range = np.max(scores) - np.min(scores)
        # Trend: positive = improving, negative = declining
        if len(scores) >= 2:
            slope, _, r_value, _, _ = stats.linregress(range(len(scores)), scores)
        else:
            slope, r_value = 0, 0

        # Late-conversation stability (last 3 turns)
        if len(scores) >= 4:
            late_std = np.std(scores[-3:])
            early_std = np.std(scores[:3])
        else:
            late_std = score_std
            early_std = score_std

        records.append({
            "model_label": model,
            "question": question[:100],
            "judgeCategory": judge_cat,
            "eval_type": "Subjective" if judge_cat == group["questionCategory"].iloc[0] else "Tangential",
            "score_mean": round(np.mean(scores), 2),
            "score_std": round(score_std, 2),
            "score_range": round(score_range, 2),
            "trend_slope": round(slope, 3),
            "trend_r2": round(r_value ** 2, 3),
            "early_volatility": round(early_std, 2),
            "late_volatility": round(late_std, 2),
            "stability_improvement": round(early_std - late_std, 2),
            "first_score": scores[0],
            "last_score": scores[-1],
            "n_turns": len(scores),
        })

    return pd.DataFrame(records)


def drift_summary_by_model(drift_df: pd.DataFrame) -> pd.DataFrame:
    """Summary of contextual drift per model."""
    if drift_df.empty:
        return pd.DataFrame()
    return drift_df.groupby("model_label").agg(
        avg_volatility=("score_std", "mean"),
        avg_range=("score_range", "mean"),
        avg_trend=("trend_slope", "mean"),
        avg_late_volatility=("late_volatility", "mean"),
        avg_stability_improvement=("stability_improvement", "mean"),
        pct_declining=("trend_slope", lambda x: (x < -0.5).mean() * 100),
    ).reset_index().sort_values("avg_volatility")


# ======================================================================
# 4. BIAS & FAIRNESS ANALYSIS
# ======================================================================

def persona_bias_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze whether certain question dimensions (which map to personas)
    receive systematically different scores.
    """
    records = []
    for model, model_df in df.groupby("model_label"):
        overall_mean = model_df["alignment_score"].mean()
        for dim in get_dimension_names():
            dim_df = model_df[model_df["questionCategory"] == dim]
            if dim_df.empty:
                continue
            dim_mean = dim_df["alignment_score"].mean()
            dim_std = dim_df["alignment_score"].std()
            n = len(dim_df)

            # Statistical test: is this dimension's mean significantly different?
            if n > 10 and dim_std > 0:
                t_stat, p_value = stats.ttest_1samp(dim_df["alignment_score"], overall_mean)
            else:
                t_stat, p_value = 0, 1.0

            records.append({
                "model_label": model,
                "dimension": dim,
                "mean_score": round(dim_mean, 2),
                "overall_mean": round(overall_mean, 2),
                "deviation": round(dim_mean - overall_mean, 2),
                "std": round(dim_std, 2),
                "n_samples": n,
                "t_stat": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
            })
    return pd.DataFrame(records)


def conversation_length_fairness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze whether conversation length (depth) systematically
    advantages or disadvantages certain models.
    """
    records = []
    for model, model_df in df.groupby("model_label"):
        for depth in sorted(model_df["depth"].unique()):
            depth_df = model_df[model_df["depth"] == depth]
            records.append({
                "model_label": model,
                "depth": depth,
                "mean_score": round(depth_df["alignment_score"].mean(), 2),
                "std_score": round(depth_df["alignment_score"].std(), 2),
                "n_samples": len(depth_df),
            })
    return pd.DataFrame(records)


def judge_model_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if different judge models produce consistent scores.
    (If only one judge model is used, this shows consistency across dimensions.)
    """
    records = []
    for model, model_df in df.groupby("model_label"):
        for judge, judge_df in model_df.groupby("judgeModel"):
            records.append({
                "model_label": model,
                "judge_model": judge,
                "mean_score": round(judge_df["alignment_score"].mean(), 2),
                "std_score": round(judge_df["alignment_score"].std(), 2),
                "n_samples": len(judge_df),
            })
    return pd.DataFrame(records)


def question_difficulty_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Identify which questions are consistently hard or easy across models."""
    records = []
    for question, q_df in df.groupby("question"):
        # Only look at subjective evaluations
        subj = q_df[q_df["questionCategory"] == q_df["judgeCategory"]]
        if subj.empty:
            continue
        records.append({
            "question": question[:120],
            "dimension": subj["questionCategory"].iloc[0],
            "mean_score": round(subj["alignment_score"].mean(), 2),
            "std_score": round(subj["alignment_score"].std(), 2),
            "min_score": subj["alignment_score"].min(),
            "max_score": subj["alignment_score"].max(),
            "score_range": subj["alignment_score"].max() - subj["alignment_score"].min(),
            "n_models": subj["model_label"].nunique(),
            "n_evaluations": len(subj),
        })
    result = pd.DataFrame(records)
    return result.sort_values("mean_score") if not result.empty else result


# ======================================================================
# 5. RUBRIC WEIGHT VALIDATION
# ======================================================================

def rubric_weight_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze whether rubric weights correspond to actual score impact.
    For each rubric question, compute the correlation between pass/fail
    and the alignment score.
    """
    records = []
    for qnum, info in RUBRIC_QUESTIONS.items():
        col = f"q{qnum}_pass"
        if col not in df.columns:
            continue

        valid = df[[col, "alignment_score"]].dropna()
        if len(valid) < 100:
            continue

        pass_scores = valid[valid[col] == True]["alignment_score"]
        fail_scores = valid[valid[col] == False]["alignment_score"]

        if len(pass_scores) < 10 or len(fail_scores) < 10:
            actual_impact = 0
            t_stat, p_value = 0, 1.0
        else:
            actual_impact = pass_scores.mean() - fail_scores.mean()
            t_stat, p_value = stats.ttest_ind(pass_scores, fail_scores)

        # Point-biserial correlation
        if valid[col].std() > 0:
            corr, corr_p = stats.pointbiserialr(valid[col].astype(int), valid["alignment_score"])
        else:
            corr, corr_p = 0, 1.0

        records.append({
            "question_num": qnum,
            "question_text": info["text"][:80],
            "cluster": info["cluster"],
            "assigned_weight": info["weight"],
            "pass_rate_pct": round(valid[col].mean() * 100, 1),
            "avg_score_when_pass": round(pass_scores.mean(), 2) if len(pass_scores) > 0 else None,
            "avg_score_when_fail": round(fail_scores.mean(), 2) if len(fail_scores) > 0 else None,
            "actual_impact": round(actual_impact, 2),
            "correlation": round(corr, 3),
            "correlation_p": round(corr_p, 4),
            "weight_impact_ratio": round(info["weight"] / max(abs(actual_impact), 0.01), 2) if info["weight"] != 0 else 0,
            "t_stat": round(t_stat, 3),
            "significant": p_value < 0.05,
        })

    result = pd.DataFrame(records)
    return result.sort_values("actual_impact", ascending=False) if not result.empty else result


def rubric_cluster_importance(weight_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weight validation by rubric cluster."""
    if weight_df.empty:
        return pd.DataFrame()
    return weight_df.groupby("cluster").agg(
        total_assigned_weight=("assigned_weight", "sum"),
        avg_actual_impact=("actual_impact", "mean"),
        avg_correlation=("correlation", "mean"),
        n_questions=("question_num", "count"),
    ).reset_index().sort_values("avg_actual_impact", ascending=False)


# ======================================================================
# 6. DATA PROVENANCE
# ======================================================================

def data_provenance_summary(df: pd.DataFrame) -> dict:
    """Generate a data provenance summary."""
    return {
        "total_rows": len(df),
        "models_tested": sorted(df["model_label"].unique().tolist()),
        "n_models": df["model_label"].nunique(),
        "judge_models": sorted(df["judgeModel"].unique().tolist()) if "judgeModel" in df.columns else [],
        "proxy_models": sorted(df["proxyModel"].unique().tolist()) if "proxyModel" in df.columns else [],
        "dimensions": sorted(df["questionCategory"].unique().tolist()),
        "n_dimensions": df["questionCategory"].nunique(),
        "depth_range": (int(df["depth"].min()), int(df["depth"].max())),
        "n_unique_questions": df["question"].nunique() if "question" in df.columns else 0,
        "questions_per_dimension": df.groupby("questionCategory")["question"].nunique().to_dict() if "question" in df.columns else {},
        "rows_per_model": df.groupby("model_label").size().to_dict(),
        "rows_per_depth": df.groupby("depth").size().to_dict(),
        "score_distribution": {
            "mean": round(df["alignment_score"].mean(), 2),
            "std": round(df["alignment_score"].std(), 2),
            "min": int(df["alignment_score"].min()),
            "max": int(df["alignment_score"].max()),
            "median": round(df["alignment_score"].median(), 2),
        },
    }
