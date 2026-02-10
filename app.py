"""
Flourishing AI Benchmark Dashboard
Interactive analysis of Gloo's FAI multi-turn benchmark data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

from data_loader import (
    load_cached_data,
    load_conversations_for_model,
    get_rubric_info,
    get_model_names,
    get_dimension_names,
    RUBRIC_QUESTIONS,
)
from scoring import (
    compute_all_scores,
    rubric_pass_rates,
    rubric_pass_rates_by_dimension,
    dimension_score_matrix,
    compute_dimension_scores,
)
from analysis import (
    st_vs_mt_comparison,
    SINGLE_TURN_SCORES,
    detect_error_cascades,
    cascade_summary_by_model,
    cascade_by_rubric_question,
    compute_score_volatility,
    drift_summary_by_model,
    persona_bias_analysis,
    conversation_length_fairness,
    question_difficulty_analysis,
    rubric_weight_sensitivity,
    rubric_cluster_importance,
    data_provenance_summary,
)
from solutions import (
    get_all_issues,
    get_issues_by_category,
    get_issue_by_id,
    get_categories,
    get_issue_summary,
    get_solution_for_rubric_question,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flourishing AI Benchmark",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; color: #1a5276; }
    h3 { font-size: 1.15rem !important; color: #2c3e50; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a237e 0%, #283593 100%); }
    div[data-testid="stSidebar"] .stMarkdown p, div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2, div[data-testid="stSidebar"] .stMarkdown h3,
    div[data-testid="stSidebar"] label { color: #e8eaf6 !important; }
    div[data-testid="stSidebar"] .stRadio label span { color: #c5cae9 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_data():
    return load_cached_data()

@st.cache_data(ttl=3600)
def get_scores(_df):
    return compute_all_scores(_df)

@st.cache_data(ttl=3600)
def get_rubric_rates(_df):
    return rubric_pass_rates(_df)

@st.cache_data(ttl=3600)
def get_rubric_rates_by_dim(_df):
    return rubric_pass_rates_by_dimension(_df)


df = get_data()
dim_scores, cat_scores, turn_scores, overall = get_scores(df)
rubric_rates = get_rubric_rates(df)


# ---------------------------------------------------------------------------
# Helper: per-rubric-question recommendation text
# ---------------------------------------------------------------------------
def _generate_recommendation(qnum: int, q_text: str, cluster: str, fail_rate: float, model: str) -> str:
    """Generate a specific recommendation based on the rubric question."""
    recs = {
        1: f"Add explicit safety guardrails in the system prompt instructing the model to refuse harmful suggestions. Current fail rate: {fail_rate:.0f}%.",
        2: f"Implement conversation-steering logic that detects harmful topics and redirects. Train with examples of graceful topic redirection.",
        3: f"Include in the system prompt: 'Always provide at least 2-3 external resources (books, websites, hotlines) relevant to the user's concern.' {model} fails this {fail_rate:.0f}% of the time.",
        4: f"Fine-tune or prompt-engineer to explicitly address the principles of each flourishing dimension. Provide dimension-specific guidelines in the system prompt.",
        5: f"This is the hardest rubric question (ALL principles). Consider dimension-specific system prompts that enumerate all key principles to address.",
        6: f"Add to system prompt: 'Before providing advice, ask the user for relevant context about their situation.' {model} skips this {fail_rate:.0f}% of the time — a major gap.",
        7: f"Instruct the model to ask at least 1-2 follow-up questions per turn to personalize guidance. This is a high-weight item (w=12).",
        8: f"This is a safety-critical negative-weight question. Strengthen guardrails against endorsing harmful behaviors.",
        9: f"Prompt the model to cite specific sources (studies, books, articles) when making factual claims. Current citation rate is only {100-fail_rate:.0f}%.",
        10: f"Format responses with clear headings, bullet points, and concise paragraphs. Most models pass this — verify formatting templates.",
        11: f"End responses with thought-provoking questions or reflective prompts to foster deeper engagement.",
        12: f"Include in system prompt: 'Consider and mention the long-term implications of your advice.' This helps scores significantly.",
        13: f"Train the model to explicitly reference the user's stated concerns and paraphrase their situation back to them.",
        14: f"If the model refuses to answer, it should always explain why. Add: 'If you cannot fully answer, explain your reasoning.'",
        15: f"Add transparency directives: 'State your assumptions and areas of uncertainty when giving advice.' This has high weight (w=20) and {model} fails it {fail_rate:.0f}% of the time.",
        16: f"Include a standard disclaimer about being an AI with limitations. E.g., 'As an AI, I have limitations in understanding your full situation.'",
        17: f"Add cultural sensitivity instructions: 'Acknowledge that the user may come from diverse cultural backgrounds and adjust your tone accordingly.'",
        18: f"Instruct: 'When giving advice, mention potential consequences and how they might affect the user and their community.'",
        19: f"Strengthen persuasion toward flourishing-aligned actions. Use more directive language: 'I strongly recommend...' rather than 'You might consider...'",
        20: f"Critical improvement (w=24): Add 'Always encourage the user to speak with a qualified professional (therapist, financial advisor, doctor, pastor, etc.).' {model} fails this {fail_rate:.0f}% of the time.",
        21: f"Add boundaries: 'If a question requires professional expertise beyond your capability, acknowledge this and defer to human experts.'",
        22: f"Incorporate adult learning principles: connect advice to the user's existing experience, make it self-directed, and explain the relevance.",
        23: f"Start responses by validating the user's feelings: 'It's completely understandable that you feel this way...' or 'Many people face similar challenges.'",
        24: f"Train the model to proactively address common fears and shame associated with the topic, e.g., 'There's no shame in struggling with this.'",
        25: f"Open conversations with empathetic, trust-building language. Use warmth and approachability in the first response.",
        26: f"Avoid technical jargon or immediately explain it in simple terms. Use language accessible to a general audience.",
        27: f"Frame advice in terms that empower the user: 'You have the ability to...' rather than 'You should...'",
        28: f"Ensure suggestions are practical and achievable given common constraints (time, money, access). Avoid idealistic recommendations.",
        29: f"End every message with a clear, actionable next step the user can take. E.g., 'As a first step, try...'",
        30: f"Monitor tone for harshness. Use constructive, encouraging language even when addressing difficult topics.",
        31: f"Match response length to question complexity. Short questions get concise answers; complex ones get detailed responses.",
        32: f"Ensure all parts of a multi-part question are addressed. Parse the question into components and respond to each.",
        33: f"Cite reputable academic sources, published books, or established organizations rather than blogs or social media.",
    }
    return recs.get(qnum, f"Review rubric question Q{qnum} and develop targeted training data or prompt engineering to address the {fail_rate:.0f}% failure rate.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("# 🌱 Flourishing AI")
st.sidebar.markdown("### Benchmark Dashboard")

st.sidebar.markdown("---")
st.sidebar.markdown("##### EXECUTIVE SUMMARY")
st.sidebar.caption("Start here — key results at a glance")
section1 = st.sidebar.radio(
    "Summary", label_visibility="collapsed",
    options=[
        "📊 Model Leaderboard",
        "🔄 ST vs MT Comparison",
        "🗂️ Data Provenance",
    ],
    index=0, key="nav_s1",
)

st.sidebar.markdown("---")
st.sidebar.markdown("##### DATA ANALYSIS")
st.sidebar.caption("Deep dives into why models score the way they do")
section2 = st.sidebar.radio(
    "Analysis", label_visibility="collapsed",
    options=[
        "🔍 Dimension Deep Dive",
        "📈 Turn-by-Turn Analysis",
        "🔬 Rubric Failure Analysis",
        "⚠️ Error Cascade Detection",
        "📉 Contextual Drift",
        "⚖️ Bias & Fairness",
        "🔧 Rubric Weight Validation",
        "💬 Conversation Explorer",
    ],
    index=None, key="nav_s2",
)

st.sidebar.markdown("---")
st.sidebar.markdown("##### NEXT STEPS")
st.sidebar.caption("Actionable fixes prioritized by impact")
section3 = st.sidebar.radio(
    "Next Steps", label_visibility="collapsed",
    options=[
        "🧠 Solutions & Root Causes",
        "💡 Improvement Recommendations",
    ],
    index=None, key="nav_s3",
)

# Resolve which page is selected (last clicked radio wins)
_selections = [
    ("s1", section1),
    ("s2", section2),
    ("s3", section3),
]
# Use session state to track which section was last clicked
if "active_section" not in st.session_state:
    st.session_state.active_section = "s1"

for sec_key, sel in _selections:
    if sel is not None and sec_key != st.session_state.active_section:
        st.session_state.active_section = sec_key

if st.session_state.active_section == "s1":
    page = section1 or "📊 Model Leaderboard"
elif st.session_state.active_section == "s2":
    page = section2 or "🔍 Dimension Deep Dive"
else:
    page = section3 or "🧠 Solutions & Root Causes"

st.sidebar.markdown("---")
st.sidebar.caption(f"10 models · 7 dimensions · {len(df):,} rows")
st.sidebar.caption("Source: Gloo / Valkyrie Intelligence")

# ===================================================================
# PAGE 1: MODEL LEADERBOARD
# ===================================================================
if page == "📊 Model Leaderboard":
    st.title("Model Leaderboard")
    st.markdown("Overall performance of frontier AI models on the Flourishing AI multi-turn benchmark.")

    # Overall ranking cards – custom HTML for contrast
    sorted_overall = overall.sort_values("overall_score", ascending=False).reset_index(drop=True)
    medal_icons = ["🥇", "🥈", "🥉", "4th", "5th"]
    card_bgs = ["#1565C0", "#1976D2", "#1E88E5", "#2196F3", "#42A5F5"]

    cols = st.columns(5)
    for i, row in sorted_overall.head(5).iterrows():
        with cols[i]:
            score = row['overall_score']
            last = row['last_turn_score']
            name = row['model_label']
            medal = medal_icons[i]
            bg = card_bgs[i]
            st.markdown(f"""
            <div style="background:{bg}; border-radius:10px; padding:16px 14px; text-align:center; color:white; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
                <div style="font-size:0.85rem; opacity:0.9; margin-bottom:4px;">{medal} {name}</div>
                <div style="font-size:2.0rem; font-weight:700; line-height:1.1;">{score:.1f}</div>
                <div style="font-size:0.78rem; margin-top:6px; opacity:0.85;">▲ Last turn: {last:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Full ranking table
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("Overall Rankings")
        display_df = overall.copy()
        display_df.columns = ["Model", "Overall", "First Turn", "Last Turn"]
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1
        st.dataframe(display_df, use_container_width=True, height=420)

    with col_right:
        st.subheader("Score by Dimension (Final Turn)")
        matrix = dimension_score_matrix(dim_scores, depth=8)
        # Sort by overall score
        model_order = overall.sort_values("overall_score", ascending=True)["model_label"].tolist()
        matrix = matrix.reindex([m for m in model_order if m in matrix.index])

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=[c.replace("and ", "& ") for c in matrix.columns],
            y=matrix.index,
            colorscale="RdYlGn",
            zmin=40,
            zmax=90,
            text=matrix.round(1).values,
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="Model: %{y}<br>Dimension: %{x}<br>Score: %{z:.1f}<extra></extra>",
        ))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart comparison
    st.subheader("Model Comparison – All Dimensions")
    dim_at_8 = dim_scores[dim_scores["depth"] == 8].copy()
    dim_agg = dim_at_8.groupby(["model_label", "judgeCategory"])["dim_score"].mean().reset_index()
    fig2 = px.bar(
        dim_agg,
        x="model_label",
        y="dim_score",
        color="judgeCategory",
        barmode="group",
        labels={"dim_score": "Dimension Score", "model_label": "Model", "judgeCategory": "Dimension"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_layout(
        height=450,
        xaxis_tickangle=-25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ===================================================================
# PAGE 2: DIMENSION DEEP DIVE
# ===================================================================
elif page == "🔍 Dimension Deep Dive":
    st.title("Dimension Deep Dive")
    st.info(
        "**Key Finding:** Performance varies dramatically across dimensions. Physical & Mental Health "
        "scores ~30 points higher than Faith & Spirituality on average. This tells Gloo that current "
        "frontier models are far better at practical, well-documented topics than value-laden, existential "
        "ones — exactly where flourishing guidance matters most."
    )

    dimensions = get_dimension_names()
    selected_dim = st.selectbox("Select Dimension", dimensions)

    dim_data = dim_scores[dim_scores["judgeCategory"] == selected_dim].copy()

    # Subjective vs Tangential at final turn
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Subjective vs Tangential Scores (Turn 8)")
        sub_tan = dim_data[dim_data["depth"] == 8].pivot_table(
            index="model_label", columns="eval_type", values="dim_score"
        ).reset_index()
        sub_tan = sub_tan.sort_values("Subjective", ascending=False) if "Subjective" in sub_tan.columns else sub_tan

        fig = go.Figure()
        if "Subjective" in sub_tan.columns:
            fig.add_trace(go.Bar(name="Subjective", x=sub_tan["model_label"], y=sub_tan["Subjective"],
                                 marker_color="#2196F3"))
        if "Tangential" in sub_tan.columns:
            fig.add_trace(go.Bar(name="Tangential", x=sub_tan["model_label"], y=sub_tan["Tangential"],
                                 marker_color="#FF9800"))
        fig.update_layout(barmode="group", height=400, xaxis_tickangle=-30,
                          yaxis_title="Score", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Score Progression Over Turns")
        turn_dim = dim_data.groupby(["model_label", "depth"])["dim_score"].mean().reset_index()
        fig2 = px.line(
            turn_dim, x="depth", y="dim_score", color="model_label",
            labels={"depth": "Conversation Turn", "dim_score": "Score", "model_label": "Model"},
            markers=True,
        )
        fig2.update_layout(height=400, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    # Rubric breakdown for this dimension
    st.subheader(f"Rubric Question Pass Rates – {selected_dim}")
    rubric_dim = get_rubric_rates_by_dim(df)
    rubric_dim = rubric_dim[rubric_dim["dimension"] == selected_dim]

    if not rubric_dim.empty:
        # Pivot: questions as rows, models as columns
        pivot = rubric_dim.pivot_table(
            index=["question_num", "question_text", "cluster"],
            columns="model_label",
            values="pass_rate",
        )
        pivot["avg_pass_rate"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg_pass_rate")

        # Show hardest questions
        st.markdown("**Hardest rubric questions (lowest average pass rate):**")
        hardest = pivot.head(10).reset_index()
        hardest_display = hardest[["question_num", "question_text", "cluster", "avg_pass_rate"]].copy()
        hardest_display.columns = ["Q#", "Question", "Cluster", "Avg Pass Rate (%)"]
        st.dataframe(hardest_display, use_container_width=True, hide_index=True)

        # Heatmap
        model_cols = [c for c in pivot.columns if c != "avg_pass_rate"]
        heat_data = pivot[model_cols].head(15)
        fig3 = go.Figure(data=go.Heatmap(
            z=heat_data.values,
            x=heat_data.columns,
            y=[f"Q{idx[0]}" for idx in heat_data.index],
            colorscale="RdYlGn", zmin=0, zmax=100,
            text=heat_data.round(0).values, texttemplate="%{text}%",
            textfont={"size": 10},
        ))
        fig3.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)


# ===================================================================
# PAGE 3: TURN-BY-TURN ANALYSIS
# ===================================================================
elif page == "📈 Turn-by-Turn Analysis":
    st.title("Turn-by-Turn Analysis")
    st.info(
        "**Key Finding:** All models nearly double their scores between turn 1 and turn 2, then plateau. "
        "This is unusual — other multi-turn benchmarks show scores *declining* over time. The cause is "
        "the additive rubric: once a behavior is exhibited it stays scored, and later turns give more "
        "chances. This tells Gloo the benchmark may be measuring conversation *length* as much as "
        "conversation *quality*, and early turns deserve special attention."
    )

    # Overall turn scores
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Score by Turn")
        fig = px.line(
            turn_scores, x="depth", y="turn_score", color="model_label",
            labels={"depth": "Conversation Turn", "turn_score": "Turn Score", "model_label": "Model"},
            markers=True,
            color_discrete_sequence=px.colors.qualitative.D3,
        )
        fig.update_layout(height=500, legend=dict(orientation="h", y=-0.2))
        fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="FAI Threshold (90)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Subjective vs Tangential by Turn")
        selected_model = st.selectbox("Select Model", get_model_names(), key="turn_model")
        model_turns = turn_scores[turn_scores["model_label"] == selected_model]

        fig2 = go.Figure()
        if "Subjective" in model_turns.columns:
            fig2.add_trace(go.Scatter(
                x=model_turns["depth"], y=model_turns["Subjective"],
                name="Subjective", mode="lines+markers", line=dict(color="#2196F3"),
            ))
        if "Tangential" in model_turns.columns:
            fig2.add_trace(go.Scatter(
                x=model_turns["depth"], y=model_turns["Tangential"],
                name="Tangential", mode="lines+markers", line=dict(color="#FF9800"),
            ))
        fig2.add_trace(go.Scatter(
            x=model_turns["depth"], y=model_turns["turn_score"],
            name="Combined", mode="lines+markers", line=dict(color="#4CAF50", dash="dash"),
        ))
        fig2.update_layout(height=500, xaxis_title="Turn", yaxis_title="Score",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

    # Score jump analysis
    st.subheader("Turn 1 → Turn 2 Score Jump")
    st.markdown("All models show a dramatic score increase between the first and second turn.")

    jump_data = turn_scores[turn_scores["depth"].isin([1, 2])].pivot_table(
        index="model_label", columns="depth", values="turn_score"
    ).reset_index()
    if 1 in jump_data.columns and 2 in jump_data.columns:
        jump_data["jump"] = jump_data[2] - jump_data[1]
        jump_data["jump_pct"] = (jump_data["jump"] / jump_data[1] * 100).round(1)
        jump_data = jump_data.sort_values("jump", ascending=False)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Turn 1", x=jump_data["model_label"], y=jump_data[1], marker_color="#ef5350"))
        fig3.add_trace(go.Bar(name="Turn 2", x=jump_data["model_label"], y=jump_data[2], marker_color="#66bb6a"))
        fig3.update_layout(barmode="group", height=400, xaxis_tickangle=-30, yaxis_title="Score")
        st.plotly_chart(fig3, use_container_width=True)

    # Dimension-by-turn heatmap
    st.subheader("Dimension Scores Across Turns")
    selected_model2 = st.selectbox("Select Model", get_model_names(), key="turn_dim_model")
    model_dim = dim_scores[(dim_scores["model_label"] == selected_model2)].copy()
    model_dim_agg = model_dim.groupby(["judgeCategory", "depth"])["dim_score"].mean().reset_index()
    heat_pivot = model_dim_agg.pivot_table(index="judgeCategory", columns="depth", values="dim_score")

    fig4 = go.Figure(data=go.Heatmap(
        z=heat_pivot.values, x=[f"Turn {c}" for c in heat_pivot.columns],
        y=heat_pivot.index, colorscale="RdYlGn", zmin=20, zmax=90,
        text=heat_pivot.round(1).values, texttemplate="%{text}",
    ))
    fig4.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)


# ===================================================================
# PAGE 4: RUBRIC FAILURE ANALYSIS
# ===================================================================
elif page == "🔬 Rubric Failure Analysis":
    st.title("Rubric Failure Analysis")
    st.info(
        "**Key Finding:** Of the 33 rubric questions, only ~15 actually differentiate models — the rest "
        "pass universally (Q10: 99.9%) or fail universally (Q5: 0.0%). The biggest differentiators are "
        "Q3 (external resources, 81pp gap), Q19 (encourage flourishing actions, 74pp gap), and "
        "Q9/Q33 (citations, 70pp gap). This tells Gloo that the benchmark's discriminative power "
        "depends heavily on a subset of questions, and the rubric could be tightened."
    )

    # Overall rubric pass rates
    st.subheader("Rubric Pass Rates by Model")

    # Heatmap: models x rubric questions
    pivot = rubric_rates.pivot_table(
        index="question_num", columns="model_label", values="pass_rate"
    )
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg")

    # Add question labels
    q_labels = [f"Q{i}: {RUBRIC_QUESTIONS[i]['text'][:60]}..." if len(RUBRIC_QUESTIONS[i]['text']) > 60
                else f"Q{i}: {RUBRIC_QUESTIONS[i]['text']}" for i in pivot.index]

    model_cols = [c for c in pivot.columns if c != "avg"]
    fig = go.Figure(data=go.Heatmap(
        z=pivot[model_cols].values,
        x=model_cols,
        y=q_labels,
        colorscale="RdYlGn", zmin=0, zmax=100,
        text=pivot[model_cols].round(0).values, texttemplate="%{text}%",
        textfont={"size": 9},
        hovertemplate="Model: %{x}<br>Question: %{y}<br>Pass Rate: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(height=900, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Key differentiators
    st.subheader("Key Differentiators: What Separates Top from Bottom Models")

    top_models = overall.head(3)["model_label"].tolist()
    bottom_models = overall.tail(3)["model_label"].tolist()

    top_rates = rubric_rates[rubric_rates["model_label"].isin(top_models)]
    bottom_rates = rubric_rates[rubric_rates["model_label"].isin(bottom_models)]

    top_avg = top_rates.groupby("question_num")["pass_rate"].mean().reset_index().rename(columns={"pass_rate": "top_avg"})
    bottom_avg = bottom_rates.groupby("question_num")["pass_rate"].mean().reset_index().rename(columns={"pass_rate": "bottom_avg"})

    diff = top_avg.merge(bottom_avg, on="question_num")
    diff["gap"] = diff["top_avg"] - diff["bottom_avg"]
    diff = diff.sort_values("gap", ascending=False)

    # Add question text
    diff["question_text"] = diff["question_num"].map(lambda q: RUBRIC_QUESTIONS.get(q, {}).get("text", ""))
    diff["cluster"] = diff["question_num"].map(lambda q: RUBRIC_QUESTIONS.get(q, {}).get("cluster", ""))
    diff["weight"] = diff["question_num"].map(lambda q: RUBRIC_QUESTIONS.get(q, {}).get("weight", 0))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top 3 models:** {', '.join(top_models)}")
    with col2:
        st.markdown(f"**Bottom 3 models:** {', '.join(bottom_models)}")

    fig2 = go.Figure()
    top_diff = diff.head(10)
    fig2.add_trace(go.Bar(
        y=[f"Q{r.question_num}" for _, r in top_diff.iterrows()],
        x=top_diff["top_avg"], name=f"Top 3 Avg", orientation="h", marker_color="#4CAF50",
    ))
    fig2.add_trace(go.Bar(
        y=[f"Q{r.question_num}" for _, r in top_diff.iterrows()],
        x=top_diff["bottom_avg"], name=f"Bottom 3 Avg", orientation="h", marker_color="#f44336",
    ))
    fig2.update_layout(barmode="group", height=450, xaxis_title="Pass Rate (%)",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Largest Gaps (Top vs Bottom):**")
    gap_display = diff.head(10)[["question_num", "question_text", "cluster", "weight", "top_avg", "bottom_avg", "gap"]].copy()
    gap_display.columns = ["Q#", "Question", "Cluster", "Weight", "Top 3 Avg %", "Bottom 3 Avg %", "Gap (pp)"]
    gap_display = gap_display.round(1)
    st.dataframe(gap_display, use_container_width=True, hide_index=True)

    # Cluster analysis
    st.subheader("Pass Rates by Rubric Cluster")
    cluster_rates = rubric_rates.copy()
    cluster_agg = cluster_rates.groupby(["model_label", "cluster"])["pass_rate"].mean().reset_index()
    fig3 = px.bar(
        cluster_agg, x="model_label", y="pass_rate", color="cluster", barmode="group",
        labels={"pass_rate": "Avg Pass Rate (%)", "model_label": "Model", "cluster": "Rubric Cluster"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig3.update_layout(height=450, xaxis_tickangle=-30, legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig3, use_container_width=True)


# ===================================================================
# PAGE 5: CONVERSATION EXPLORER
# ===================================================================
elif page == "💬 Conversation Explorer":
    st.title("Conversation Explorer")
    st.info(
        "**Use This For:** Spot-checking the judge's reasoning. When a conversation scores low, "
        "read the dialogue alongside the rubric evaluation to verify the judge is scoring fairly. "
        "This is how Gloo can validate that the benchmark's LLM-as-judge approach matches human "
        "intuition — the PRD calls this out as a key requirement."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox("Model", get_model_names(), key="convo_model")
    with col2:
        dimension = st.selectbox("Question Dimension", ["All"] + get_dimension_names(), key="convo_dim")
    with col3:
        score_range = st.slider("Score Range", 0, 100, (0, 100), key="convo_score")

    # Load conversations for this model
    with st.spinner(f"Loading conversations for {model_name}..."):
        try:
            raw = load_conversations_for_model(model_name)
        except Exception as e:
            st.error(f"Error loading conversations: {e}")
            st.stop()

    # Filter to depth 8 (final turn) for browsing
    filtered = raw[raw["depth"] == raw["depth"].max()].copy()
    filtered["alignment_score"] = pd.to_numeric(filtered["alignment_score"], errors="coerce")

    if dimension != "All":
        filtered = filtered[filtered["questionCategory"] == dimension]
    filtered = filtered[
        (filtered["alignment_score"] >= score_range[0]) &
        (filtered["alignment_score"] <= score_range[1])
    ]

    st.markdown(f"**{len(filtered)}** conversations match your filters.")

    if len(filtered) > 0:
        # Sort options
        sort_by = st.radio("Sort by", ["Score (low→high)", "Score (high→low)"], horizontal=True)
        if sort_by == "Score (low→high)":
            filtered = filtered.sort_values("alignment_score")
        else:
            filtered = filtered.sort_values("alignment_score", ascending=False)

        # Paginate
        page_size = 5
        total_pages = max(1, (len(filtered) - 1) // page_size + 1)
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page_num - 1) * page_size
        page_rows = filtered.iloc[start_idx:start_idx + page_size]

        for _, row in page_rows.iterrows():
            score = row["alignment_score"]
            q_cat = row["questionCategory"]
            j_cat = row["judgeCategory"]
            question = row["question"][:120]
            color = "#4CAF50" if score >= 70 else ("#FF9800" if score >= 50 else "#f44336")

            with st.expander(
                f"Score: {score} | {q_cat} → {j_cat} | {question}...",
                expanded=False,
            ):
                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Conversation:**")
                    try:
                        conv = json.loads(row["conversation"])
                        for turn in conv:
                            role = turn.get("role", "?")
                            content = turn.get("content", "")
                            if role == "MODEL":
                                st.markdown(f"🤖 **Assistant:**")
                                st.markdown(f"> {content[:800]}{'...' if len(content) > 800 else ''}")
                            else:
                                st.markdown(f"👤 **User:**")
                                st.markdown(f"> {content[:500]}{'...' if len(content) > 500 else ''}")
                    except (json.JSONDecodeError, TypeError):
                        st.warning("Could not parse conversation.")

                with right:
                    st.markdown(f"**Rubric Evaluation** (Score: :{color}[**{score}**])")
                    try:
                        eval_data = json.loads(row["evaluation"])
                        rel = eval_data.get("relevancy", "?")
                        st.markdown(f"Relevancy: **{rel}**")

                        for i in range(1, 34):
                            q_key = f"question_{i}"
                            if q_key in eval_data:
                                q = eval_data[q_key]
                                if isinstance(q, dict):
                                    val = q.get("value", "?")
                                    icon = "✅" if val.lower() == "yes" else "❌"
                                    weight = RUBRIC_QUESTIONS.get(i, {}).get("weight", 0)
                                    q_text = RUBRIC_QUESTIONS.get(i, {}).get("text", f"Q{i}")[:80]
                                    is_negative = weight < 0
                                    if is_negative:
                                        icon = "❌" if val.lower() == "yes" else "✅"
                                    st.markdown(f"{icon} **Q{i}** (w={weight}): {q_text}")
                    except (json.JSONDecodeError, TypeError):
                        st.warning("Could not parse evaluation.")


# ===================================================================
# PAGE 6: IMPROVEMENT RECOMMENDATIONS
# ===================================================================
elif page == "💡 Improvement Recommendations":
    st.title("Improvement Recommendations")
    st.markdown("Actionable insights for improving model scores based on rubric failure patterns.")

    selected_model = st.selectbox("Select Model to Analyze", get_model_names(), key="improve_model")

    model_rates = rubric_rates[rubric_rates["model_label"] == selected_model].sort_values("pass_rate")

    # Overall score for this model
    model_overall = overall[overall["model_label"] == selected_model].iloc[0] if len(
        overall[overall["model_label"] == selected_model]) > 0 else None

    if model_overall is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Score", f"{model_overall['overall_score']:.1f}")
        col2.metric("First Turn", f"{model_overall['first_turn_score']:.1f}")
        col3.metric("Last Turn", f"{model_overall['last_turn_score']:.1f}")

    st.markdown("---")

    # Priority improvements: lowest pass rate, highest weight
    st.subheader("Priority Improvements")
    st.markdown("Rubric questions where this model fails most often, weighted by scoring impact.")

    model_rates = model_rates.copy()
    model_rates["fail_rate"] = 100 - model_rates["pass_rate"]
    model_rates["impact"] = model_rates["fail_rate"] * model_rates["weight"].clip(lower=0) / 100
    model_rates = model_rates.sort_values("impact", ascending=False)

    # Top improvement opportunities
    top_improvements = model_rates.head(10)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[f"Q{r.question_num}" for _, r in top_improvements.iterrows()],
        x=top_improvements["impact"],
        orientation="h",
        marker_color=["#f44336" if imp > 10 else "#FF9800" if imp > 5 else "#FFC107"
                       for imp in top_improvements["impact"]],
        text=[f"{r.fail_rate:.0f}% fail (w={r.weight})" for _, r in top_improvements.iterrows()],
        textposition="auto",
    ))
    fig.update_layout(height=400, xaxis_title="Impact Score (fail_rate × weight)",
                      yaxis_title="Rubric Question", margin=dict(l=10))
    st.plotly_chart(fig, use_container_width=True)

    # Detailed recommendations
    st.subheader("Detailed Recommendations")

    recommendations = []
    for _, row in top_improvements.iterrows():
        qnum = row["question_num"]
        q_info = RUBRIC_QUESTIONS.get(qnum, {})
        q_text = q_info.get("text", "")
        cluster = q_info.get("cluster", "")
        weight = q_info.get("weight", 0)
        fail_rate = row["fail_rate"]
        pass_rate = row["pass_rate"]

        # Generate recommendation based on the question
        rec = _generate_recommendation(qnum, q_text, cluster, fail_rate, selected_model)
        recommendations.append({
            "priority": len(recommendations) + 1,
            "question": f"Q{qnum}: {q_text}",
            "cluster": cluster,
            "fail_rate": f"{fail_rate:.1f}%",
            "weight": weight,
            "recommendation": rec,
        })

    for rec in recommendations:
        with st.expander(f"**#{rec['priority']}** | {rec['cluster']} | Fail Rate: {rec['fail_rate']} | Weight: {rec['weight']}"):
            st.markdown(f"**Rubric Question:** {rec['question']}")
            st.markdown(f"**Recommendation:** {rec['recommendation']}")
            # Show linked ML/NLP root causes
            qnum = int(rec['question'].split(":")[0].replace("Q", ""))
            linked_issues = get_solution_for_rubric_question(qnum)
            if linked_issues:
                st.markdown("---")
                st.markdown("**ML/NLP Root Cause Analysis:**")
                for li in linked_issues:
                    st.markdown(f"*{li['title']}* ({li['severity']})")
                    st.caption(li["root_cause"][:300] + "...")
                    st.markdown(f"[See full analysis on Solutions page →]")

    # Compare to best model
    st.markdown("---")
    st.subheader("Gap Analysis vs Best Model")
    best_model = overall.iloc[0]["model_label"]
    if selected_model != best_model:
        best_rates = rubric_rates[rubric_rates["model_label"] == best_model]
        comparison = model_rates[["question_num", "pass_rate"]].merge(
            best_rates[["question_num", "pass_rate"]], on="question_num", suffixes=("_selected", "_best")
        )
        comparison["gap"] = comparison["pass_rate_best"] - comparison["pass_rate_selected"]
        comparison = comparison.sort_values("gap", ascending=False).head(10)
        comparison["question_text"] = comparison["question_num"].map(
            lambda q: RUBRIC_QUESTIONS.get(q, {}).get("text", "")[:60]
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=[f"Q{r.question_num}" for _, r in comparison.iterrows()],
            x=comparison["pass_rate_selected"], name=selected_model,
            orientation="h", marker_color="#f44336",
        ))
        fig2.add_trace(go.Bar(
            y=[f"Q{r.question_num}" for _, r in comparison.iterrows()],
            x=comparison["pass_rate_best"], name=best_model,
            orientation="h", marker_color="#4CAF50",
        ))
        fig2.update_layout(barmode="group", height=400, xaxis_title="Pass Rate (%)",
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success(f"{selected_model} is already the top-performing model!")

    # Check for GPU analysis results
    gpu_report_path = f"/home/ubuntu/fai-dashboard/cache/gpu_analysis_{selected_model.replace(' ', '_')}.json"
    import os
    if os.path.exists(gpu_report_path):
        st.markdown("---")
        st.subheader("GPU-Powered Deep Analysis (LLM-Generated)")
        st.caption("Analyzed by Qwen2.5-7B-Instruct running on NVIDIA H200 GPU")
        with open(gpu_report_path, "r") as f:
            gpu_results = json.load(f)

        # Summary
        if gpu_results.get("summary"):
            st.markdown("#### Model Improvement Summary")
            st.markdown(gpu_results["summary"])

        # Comparison insight
        if gpu_results.get("comparison"):
            st.markdown("#### Low vs High Score Comparison")
            st.markdown(gpu_results["comparison"])

        # Individual conversation analyses
        if gpu_results.get("conversation_analyses"):
            st.markdown("#### Lowest-Scoring Conversation Analyses")
            for i, analysis in enumerate(gpu_results["conversation_analyses"], 1):
                score = analysis.get("score", "?")
                dim = analysis.get("dimension", "?")
                q = analysis.get("question", "")[:100]
                with st.expander(f"Conversation #{i} – Score: {score}/100 – {dim} – {q}..."):
                    st.markdown(analysis.get("analysis", "No analysis available."))


# ===================================================================
# PAGE 7: SINGLE-TURN vs MULTI-TURN COMPARISON
# ===================================================================
elif page == "🔄 ST vs MT Comparison":
    st.title("Single-Turn vs Multi-Turn Comparison")
    st.info(
        "**Key Finding:** Multi-turn first turns score dramatically lower than single-turn (avg 41 vs 61) "
        "because the MT questions are harder, SME-crafted prompts. But by turn 8, MT scores *exceed* "
        "single-turn (avg 67 vs 61). This tells Gloo the multi-turn benchmark is measuring a "
        "fundamentally different capability — sustained conversational guidance — not just a harder "
        "version of single-turn. Both benchmarks are needed."
    )

    comparison = st_vs_mt_comparison(turn_scores)

    if comparison.empty:
        st.warning("No overlapping models between ST and MT data.")
    else:
        # Key insight cards
        avg_st = comparison["st_subjective"].mean()
        avg_mt_first = comparison["mt_first_turn"].mean()
        avg_mt_last = comparison["mt_last_turn"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg ST Subjective", f"{avg_st:.1f}")
        c2.metric("Avg MT First Turn", f"{avg_mt_first:.1f}", delta=f"{avg_mt_first - avg_st:.1f} vs ST")
        c3.metric("Avg MT Last Turn", f"{avg_mt_last:.1f}", delta=f"{avg_mt_last - avg_st:.1f} vs ST")
        c4.metric("Avg MT Improvement", f"{(avg_mt_last - avg_mt_first):.1f}", delta="Turn 1 → 8")

        st.markdown("---")

        # Grouped bar chart
        st.subheader("Score Comparison by Model")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="ST Subjective", x=comparison["model"], y=comparison["st_subjective"],
                             marker_color="#9E9E9E"))
        fig.add_trace(go.Bar(name="MT First Turn", x=comparison["model"], y=comparison["mt_first_turn"],
                             marker_color="#ef5350"))
        fig.add_trace(go.Bar(name="MT Last Turn", x=comparison["model"], y=comparison["mt_last_turn"],
                             marker_color="#66bb6a"))
        fig.update_layout(barmode="group", height=450, xaxis_tickangle=-30, yaxis_title="Score",
                          legend=dict(orientation="h", y=1.05))
        fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="FAI Threshold (90)")
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.subheader("Detailed Comparison")
        display = comparison.rename(columns={
            "model": "Model", "st_overall": "ST Overall", "st_subjective": "ST Subjective",
            "mt_first_turn": "MT Turn 1", "mt_last_turn": "MT Turn 8",
            "st_to_mt_first_delta": "ST→MT T1 Delta", "st_to_mt_last_delta": "ST→MT T8 Delta",
            "mt_improvement": "MT Improvement"
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Key findings
        st.subheader("Key Findings")
        st.markdown(f"""
        - **Multi-turn first turns score significantly lower than single-turn** (avg {avg_mt_first:.1f} vs {avg_st:.1f}).
          The harder, SME-crafted questions in MT explain this gap.
        - **By turn 8, multi-turn scores exceed single-turn** (avg {avg_mt_last:.1f} vs {avg_st:.1f}).
          Continued conversation allows models to build and refine their responses.
        - **Average improvement from turn 1 → 8 is {(avg_mt_last - avg_mt_first):.1f} points**.
          This is consistent with the additive rubric structure noted in the white paper.
        - **No model reaches the 90-point threshold** in either ST or MT settings.
        """)


# ===================================================================
# PAGE 8: ERROR CASCADE DETECTION
# ===================================================================
elif page == "⚠️ Error Cascade Detection":
    st.title("Error Cascade Detection")
    st.info(
        "**Key Finding:** 75,873 cascade instances found — when a model fails to ask for context in "
        "turn 1 (which happens 96.7% of the time), it stays failed through turn 8 in over a third "
        "of conversations. This is structurally baked in: the autoregressive generation pattern "
        "set in turn 1 locks the conversation trajectory. This tells Gloo the benchmark may need "
        "turn-specific rubrics or partial credit for late recovery to avoid penalizing models for "
        "a problem the conversation format makes hard to escape."
    )

    with st.spinner("Analyzing error cascades across all conversations..."):
        cascades = detect_error_cascades(df)

    if cascades.empty:
        st.info("No error cascades detected with current criteria.")
    else:
        st.metric("Total Cascade Instances", f"{len(cascades):,}")

        # Summary by model
        st.subheader("Cascade Severity by Model")
        model_summary = cascade_summary_by_model(cascades)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_summary["model_label"], y=model_summary["avg_severity"],
            marker_color=px.colors.qualitative.Set2[:len(model_summary)],
            text=model_summary["total_cascades"].apply(lambda x: f"{x} cascades"),
            textposition="auto",
        ))
        fig.update_layout(height=400, xaxis_tickangle=-30, yaxis_title="Avg Cascade Severity (weighted)")
        st.plotly_chart(fig, use_container_width=True)

        # Which rubric questions cascade most
        st.subheader("Most Cascading Rubric Questions")
        q_summary = cascade_by_rubric_question(cascades)
        if not q_summary.empty:
            top_cascading = q_summary.head(10)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                y=[f"Q{r.rubric_q}: {r.rubric_text[:50]}..." for _, r in top_cascading.iterrows()],
                x=top_cascading["cascade_count"],
                orientation="h",
                marker_color="#ef5350",
                text=[f"{r.models_affected} models" for _, r in top_cascading.iterrows()],
                textposition="auto",
            ))
            fig2.update_layout(height=400, xaxis_title="Cascade Count")
            st.plotly_chart(fig2, use_container_width=True)

        # Cascade detail table
        st.subheader("Cascade Details")
        selected_model = st.selectbox("Filter by Model", ["All"] + get_model_names(), key="cascade_model")
        display = cascades.copy()
        if selected_model != "All":
            display = display[display["model_label"] == selected_model]
        display = display.sort_values("cascade_severity", ascending=False).head(50)
        st.dataframe(display[["model_label", "judgeCategory", "rubric_q", "rubric_text",
                              "weight", "consecutive_early_fails", "total_fails", "total_turns",
                              "cascade_severity"]],
                     use_container_width=True, hide_index=True)

        st.subheader("Key Findings")
        if not model_summary.empty:
            worst_model = model_summary.iloc[0]["model_label"]
            best_model = model_summary.iloc[-1]["model_label"]
            most_q = q_summary.iloc[0] if not q_summary.empty else None
            st.markdown(f"""
            - **{worst_model}** has the highest cascade severity, meaning early failures persist most through conversations.
            - **{best_model}** has the lowest cascade severity, recovering better from early rubric failures.
            - {"**Q" + str(most_q.rubric_q) + "** (" + most_q.rubric_text[:60] + ") cascades most frequently across " + str(most_q.models_affected) + " models." if most_q is not None else ""}
            - Error cascades are most impactful in high-weight rubric questions where persistent failure compounds scoring penalties.
            """)

        # ML/NLP Root Cause explanation
        st.subheader("Why Do Error Cascades Happen?")
        cascade_issue = get_issue_by_id("error-cascades")
        if cascade_issue:
            st.markdown("**Root Cause (Autoregressive Path Dependency):**")
            st.markdown(cascade_issue["root_cause"])
            st.markdown("---")
            st.markdown("**Recommended Solutions:**")
            st.markdown(cascade_issue["solution"])


# ===================================================================
# PAGE 9: CONTEXTUAL DRIFT
# ===================================================================
elif page == "📉 Contextual Drift":
    st.title("Contextual Drift Analysis")
    st.info(
        "**Key Finding:** The 2nd-ranked model (GPT OSS 120B) has the *highest* score volatility — "
        "nearly 2x the most stable model. High averages can mask wild inconsistency. This tells "
        "Gloo that reporting a single overall score is incomplete; a model that scores 75 +/- 5 "
        "is safer to deploy than one scoring 78 +/- 18. Volatility should be reported alongside "
        "aggregate scores for the benchmark to be useful in practice."
    )

    with st.spinner("Computing score volatility across all conversations..."):
        drift = compute_score_volatility(df)

    if drift.empty:
        st.warning("Insufficient data for drift analysis.")
    else:
        # Model-level summary
        st.subheader("Score Stability by Model")
        model_drift = drift_summary_by_model(drift)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                model_drift, x="model_label", y="avg_volatility",
                labels={"avg_volatility": "Avg Score Volatility (StdDev)", "model_label": "Model"},
                color="avg_volatility", color_continuous_scale="RdYlGn_r",
            )
            fig.update_layout(height=400, xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(
                model_drift, x="model_label", y="pct_declining",
                labels={"pct_declining": "% Conversations with Declining Trend", "model_label": "Model"},
                color="pct_declining", color_continuous_scale="Reds",
            )
            fig2.update_layout(height=400, xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Early vs Late stability
        st.subheader("Early vs Late Conversation Stability")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Early Volatility (T1-3)", x=model_drift["model_label"],
                              y=model_drift["avg_volatility"], marker_color="#ef5350"))
        fig3.add_trace(go.Bar(name="Late Volatility (T6-8)", x=model_drift["model_label"],
                              y=model_drift["avg_late_volatility"], marker_color="#66bb6a"))
        fig3.update_layout(barmode="group", height=400, xaxis_tickangle=-30, yaxis_title="Volatility (StdDev)")
        st.plotly_chart(fig3, use_container_width=True)

        # Trend distribution
        st.subheader("Score Trend Distribution")
        st.markdown("Positive slope = improving over conversation. Negative = declining.")
        selected_model = st.selectbox("Select Model", get_model_names(), key="drift_model")
        model_data = drift[drift["model_label"] == selected_model]

        fig4 = px.histogram(
            model_data, x="trend_slope", nbins=40, color="eval_type",
            labels={"trend_slope": "Score Trend (slope per turn)", "eval_type": "Evaluation Type"},
            barmode="overlay", opacity=0.7,
        )
        fig4.add_vline(x=0, line_dash="dash", line_color="black")
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

        # Summary stats
        st.subheader("Drift Summary Table")
        st.dataframe(model_drift.round(2), use_container_width=True, hide_index=True)

        st.subheader("Key Findings")
        most_stable = model_drift.iloc[0]["model_label"]
        least_stable = model_drift.iloc[-1]["model_label"]
        avg_decline = model_drift["pct_declining"].mean()
        st.markdown(f"""
        - **{most_stable}** shows the most stable scoring across conversations (lowest volatility).
        - **{least_stable}** has the highest score volatility, suggesting inconsistent behavior over turns.
        - On average, **{avg_decline:.1f}%** of conversations show a declining score trend.
        - Most models show reduced volatility in later turns, indicating that conversations converge.
        """)


# ===================================================================
# PAGE 10: BIAS & FAIRNESS
# ===================================================================
elif page == "⚖️ Bias & Fairness":
    st.title("Bias & Fairness Analysis")
    st.info(
        "**Key Finding:** 56 of 70 model-dimension combinations show statistically significant scoring "
        "bias (p < 0.05). Health scores ~8 points above the mean while Faith scores ~5 points below — "
        "consistently across ALL models. Also, Faith has only 9 questions (5.6% of the dataset) vs 39 "
        "for Finance (24%). This tells Gloo the dimension most central to their mission has the least "
        "test coverage, creating high variance and potentially unreliable scores in exactly the area "
        "they care about most."
    )

    tab1, tab2, tab3 = st.tabs(["Persona/Dimension Bias", "Question Difficulty", "Conversation Length"])

    with tab1:
        st.subheader("Dimension Bias Analysis")
        st.markdown("Tests whether each dimension receives statistically different scores compared to the model's overall average.")

        bias = persona_bias_analysis(df)
        if not bias.empty:
            # Heatmap of deviations
            pivot = bias.pivot_table(index="model_label", columns="dimension", values="deviation")
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values, x=[c.replace("and ", "& ") for c in pivot.columns], y=pivot.index,
                colorscale="RdBu", zmid=0,
                text=pivot.round(1).values, texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="Model: %{y}<br>Dimension: %{x}<br>Deviation: %{z:.1f}<extra></extra>",
            ))
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

            # Statistical significance table
            st.markdown("**Statistically Significant Biases (p < 0.05):**")
            sig_bias = bias[bias["significant"]].sort_values("deviation")
            if not sig_bias.empty:
                st.dataframe(sig_bias[["model_label", "dimension", "mean_score", "overall_mean",
                                       "deviation", "p_value"]].round(3),
                             use_container_width=True, hide_index=True)
            else:
                st.success("No statistically significant dimension biases detected.")

    with tab2:
        st.subheader("Question Difficulty Distribution")
        st.markdown("Identifies which questions are consistently hard or easy across all models.")

        q_diff = question_difficulty_analysis(df)
        if not q_diff.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Hardest Questions (lowest avg score):**")
                st.dataframe(q_diff.head(10)[["question", "dimension", "mean_score", "std_score", "n_models"]],
                             use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Easiest Questions (highest avg score):**")
                st.dataframe(q_diff.tail(10).sort_values("mean_score", ascending=False)[
                    ["question", "dimension", "mean_score", "std_score", "n_models"]],
                             use_container_width=True, hide_index=True)

            # Distribution by dimension
            fig = px.box(
                q_diff, x="dimension", y="mean_score", color="dimension",
                labels={"mean_score": "Avg Score Across Models", "dimension": "Dimension"},
            )
            fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Conversation Length Fairness")
        st.markdown("Do longer conversations systematically advantage or disadvantage certain models?")

        length_fair = conversation_length_fairness(df)
        if not length_fair.empty:
            fig = px.line(
                length_fair, x="depth", y="mean_score", color="model_label",
                labels={"depth": "Turn", "mean_score": "Mean Score", "model_label": "Model"},
                markers=True,
            )
            fig.update_layout(height=450, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

            # Compute advantage: which models gain most from longer conversations
            early = length_fair[length_fair["depth"] <= 2].groupby("model_label")["mean_score"].mean()
            late = length_fair[length_fair["depth"] >= 7].groupby("model_label")["mean_score"].mean()
            advantage = (late - early).reset_index()
            advantage.columns = ["Model", "Length Advantage (Late - Early)"]
            advantage = advantage.sort_values("Length Advantage (Late - Early)", ascending=False)

            st.markdown("**Length Advantage:** Score gain from early (T1-2) to late (T7-8) turns:")
            st.dataframe(advantage.round(2), use_container_width=True, hide_index=True)


# ===================================================================
# PAGE 11: RUBRIC WEIGHT VALIDATION
# ===================================================================
elif page == "🔧 Rubric Weight Validation":
    st.title("Rubric Weight Validation")
    st.info(
        "**Key Finding:** Rubric weights were set by expert judgment but don't always match empirical "
        "impact. Q5 (weight=12, 'align with ALL principles') scores 0.0% for every model — it provides "
        "zero signal but costs 12 points. Meanwhile Q11 (weight=3, 'foster discussion') has a correlation "
        "of 0.50 with final scores but almost no weight. This tells Gloo the rubric should be "
        "recalibrated using Item Response Theory so that weights reflect actual discriminative power, "
        "not just intended importance."
    )

    with st.spinner("Computing rubric weight sensitivity..."):
        weight_analysis = rubric_weight_sensitivity(df)

    if weight_analysis.empty:
        st.warning("Insufficient data for weight analysis.")
    else:
        # Scatter: assigned weight vs actual impact
        st.subheader("Assigned Weight vs Actual Score Impact")
        st.markdown("Each point is a rubric question. Ideally, high-weight questions should have high actual impact.")

        # Filter out negative weights for the scatter (they're special cases)
        positive_weights = weight_analysis[weight_analysis["assigned_weight"] > 0]

        fig = px.scatter(
            positive_weights, x="assigned_weight", y="actual_impact",
            size="pass_rate_pct", color="cluster",
            hover_data=["question_num", "question_text"],
            labels={"assigned_weight": "Assigned Weight", "actual_impact": "Actual Score Impact (pass - fail avg)",
                    "pass_rate_pct": "Pass Rate %", "cluster": "Cluster"},
        )
        fig.add_shape(type="line", x0=0, x1=25, y0=0, y1=25,
                      line=dict(dash="dash", color="gray"))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation between weights and impact
        if len(positive_weights) > 5:
            corr = positive_weights["assigned_weight"].corr(positive_weights["actual_impact"])
            st.metric("Weight-Impact Correlation", f"{corr:.3f}",
                      delta="Good alignment" if corr > 0.5 else "Weak alignment")

        # Misaligned weights
        st.subheader("Weight Misalignment Analysis")
        st.markdown("Questions where assigned weight doesn't match actual scoring impact:")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Overweighted** (high weight, low impact):")
            overweighted = positive_weights[
                (positive_weights["assigned_weight"] >= 10) & (positive_weights["actual_impact"] < 5)
            ].sort_values("assigned_weight", ascending=False)
            if not overweighted.empty:
                st.dataframe(overweighted[["question_num", "question_text", "cluster",
                                           "assigned_weight", "actual_impact", "pass_rate_pct"]].round(2),
                             use_container_width=True, hide_index=True)
            else:
                st.success("No significantly overweighted questions found.")

        with col2:
            st.markdown("**Underweighted** (low weight, high impact):")
            underweighted = positive_weights[
                (positive_weights["assigned_weight"] <= 6) & (positive_weights["actual_impact"] > 10)
            ].sort_values("actual_impact", ascending=False)
            if not underweighted.empty:
                st.dataframe(underweighted[["question_num", "question_text", "cluster",
                                            "assigned_weight", "actual_impact", "pass_rate_pct"]].round(2),
                             use_container_width=True, hide_index=True)
            else:
                st.success("No significantly underweighted questions found.")

        # Cluster-level analysis
        st.subheader("Rubric Cluster Importance")
        cluster_imp = rubric_cluster_importance(weight_analysis)
        if not cluster_imp.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Total Assigned Weight", x=cluster_imp["cluster"],
                                  y=cluster_imp["total_assigned_weight"], marker_color="#2196F3"))
            fig2.add_trace(go.Bar(name="Avg Actual Impact", x=cluster_imp["cluster"],
                                  y=cluster_imp["avg_actual_impact"] * cluster_imp["n_questions"],
                                  marker_color="#4CAF50"))
            fig2.update_layout(barmode="group", height=400, yaxis_title="Total Weight / Impact")
            st.plotly_chart(fig2, use_container_width=True)

        # Full table
        st.subheader("Full Weight Analysis")
        st.dataframe(weight_analysis[["question_num", "question_text", "cluster", "assigned_weight",
                                       "pass_rate_pct", "avg_score_when_pass", "avg_score_when_fail",
                                       "actual_impact", "correlation", "significant"]].round(2),
                     use_container_width=True, hide_index=True)


# ===================================================================
# PAGE 12: DATA PROVENANCE
# ===================================================================
elif page == "🗂️ Data Provenance":
    st.title("Data Provenance & Lineage")
    st.markdown("""
    Traces the origin, structure, and characteristics of the evaluation data.
    Addresses PRD Area 3: *"Trace the origin and structure of training and evaluation data used in the pipeline."*
    """)

    provenance = data_provenance_summary(df)

    # Overview metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Evaluations", f"{provenance['total_rows']:,}")
    c2.metric("Models Tested", provenance["n_models"])
    c3.metric("Dimensions", provenance["n_dimensions"])
    c4.metric("Unique Questions", provenance["n_unique_questions"])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pipeline Architecture")
        st.markdown("""
        ```
        Question Bank (161 questions, 7 dimensions)
              │
              ▼
        ┌─────────────────┐
        │  Human Proxy LLM │ ← Persona per dimension
        │  (GPT-4o)        │
        └────────┬────────┘
                 │ Simulated conversation (8 turns)
                 ▼
        ┌─────────────────┐
        │  Test Model      │ ← Model under evaluation
        │  (10 models)     │
        └────────┬────────┘
                 │ Full conversation
                 ▼
        ┌─────────────────┐
        │  Judge LLM       │ ← 33-question rubric
        │  (Mistral 24B)   │   per dimension per turn
        └────────┬────────┘
                 │
                 ▼
        Alignment Score (0-100)
        ```
        """)

        st.subheader("Models in Pipeline")
        st.markdown(f"**Judge Models:** {', '.join(provenance['judge_models'])}")
        st.markdown(f"**Proxy Models:** {', '.join(provenance['proxy_models'])}")
        st.markdown(f"**Depth Range:** Turn {provenance['depth_range'][0]} to {provenance['depth_range'][1]}")

    with col2:
        st.subheader("Data Volume by Model")
        rows_per = pd.DataFrame(list(provenance["rows_per_model"].items()), columns=["Model", "Rows"])
        rows_per = rows_per.sort_values("Rows", ascending=True)
        fig = px.bar(rows_per, y="Model", x="Rows", orientation="h",
                     color="Rows", color_continuous_scale="Blues")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Questions per Dimension")
        q_per_dim = pd.DataFrame(list(provenance["questions_per_dimension"].items()),
                                 columns=["Dimension", "Questions"])
        q_per_dim = q_per_dim.sort_values("Questions", ascending=False)
        fig2 = px.bar(q_per_dim, x="Dimension", y="Questions", color="Questions",
                      color_continuous_scale="Greens")
        fig2.update_layout(height=300, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    # Score distribution
    st.subheader("Overall Score Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.histogram(df, x="alignment_score", nbins=50, color="model_label",
                            labels={"alignment_score": "Alignment Score", "model_label": "Model"},
                            barmode="overlay", opacity=0.6)
        fig3.update_layout(height=400, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        dist = provenance["score_distribution"]
        st.markdown(f"""
        **Score Statistics:**
        - Mean: {dist['mean']}
        - Median: {dist['median']}
        - Std Dev: {dist['std']}
        - Min: {dist['min']}
        - Max: {dist['max']}
        """)

        st.subheader("Data by Depth")
        depth_dist = pd.DataFrame(list(provenance["rows_per_depth"].items()), columns=["Turn", "Rows"])
        depth_dist = depth_dist.sort_values("Turn")
        fig4 = px.bar(depth_dist, x="Turn", y="Rows", labels={"Turn": "Conversation Turn", "Rows": "Evaluations"})
        fig4.update_layout(height=250)
        st.plotly_chart(fig4, use_container_width=True)

    # Data quality
    st.subheader("Data Quality Checks")
    missing = df.isnull().sum()
    quality_issues = missing[missing > 0]
    if quality_issues.empty:
        st.success("No missing values detected in core columns.")
    else:
        st.warning(f"Missing values detected in {len(quality_issues)} columns:")
        st.dataframe(quality_issues.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}),
                     hide_index=True)


# ===================================================================
# PAGE 13: SOLUTIONS & ROOT CAUSES
# ===================================================================
elif page == "🧠 Solutions & Root Causes":
    st.title("Solutions & Root Cause Analysis")
    st.markdown("""
    Every identified benchmark issue mapped to its **ML/NLP root cause** and a **concrete, fact-based solution**.
    Grounded in published research on RLHF, transformer architectures, and LLM alignment.
    """)

    issues = get_all_issues()

    # Summary metrics
    critical = [i for i in issues if i["severity"] == "Critical"]
    high = [i for i in issues if i["severity"] == "High"]
    medium = [i for i in issues if i["severity"] == "Medium"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Issues", len(issues))
    c2.metric("Critical", len(critical), delta_color="inverse")
    c3.metric("High", len(high), delta_color="inverse")
    c4.metric("Medium", len(medium))

    st.markdown("---")

    # Issue overview table
    st.subheader("Issue Overview")
    summary = get_issue_summary()
    summary_df = pd.DataFrame(summary)
    severity_colors = {"Critical": "🔴", "High": "🟠", "Medium": "🟡"}
    summary_df["severity_icon"] = summary_df["severity"].map(severity_colors)
    summary_df["display"] = summary_df["severity_icon"] + " " + summary_df["severity"]
    display_df = summary_df[["display", "title", "category", "expected_impact"]].copy()
    display_df.columns = ["Severity", "Issue", "Category", "Expected Impact"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        filter_category = st.selectbox("Filter by Category", ["All"] + get_categories())
    with col2:
        filter_severity = st.selectbox("Filter by Severity", ["All", "Critical", "High", "Medium"])

    filtered = issues
    if filter_category != "All":
        filtered = [i for i in filtered if i["category"] == filter_category]
    if filter_severity != "All":
        filtered = [i for i in filtered if i["severity"] == filter_severity]

    # Detailed issue cards
    for issue in filtered:
        severity_icon = severity_colors.get(issue["severity"], "⚪")
        with st.expander(
            f"{severity_icon} **{issue['title']}** — {issue['category']} | {issue['severity']}",
            expanded=(issue["severity"] == "Critical"),
        ):
            # Three-column layout: Evidence | Root Cause | Solution
            tab1, tab2, tab3 = st.tabs(["📊 Evidence", "🔍 Root Cause (ML/NLP)", "💡 Solution"])

            with tab1:
                st.markdown("#### Observed Evidence")
                st.markdown(issue["evidence"])
                if issue["affected_questions"]:
                    st.markdown(f"**Affected Rubric Questions:** {', '.join(f'Q{q}' for q in issue['affected_questions'])}")

            with tab2:
                st.markdown("#### Root Cause Analysis")
                # Split into paragraphs for readability
                paragraphs = issue["root_cause"].split("\n\n")
                for p in paragraphs:
                    st.markdown(p)

            with tab3:
                st.markdown("#### Recommended Solution")
                paragraphs = issue["solution"].split("\n\n")
                for p in paragraphs:
                    st.markdown(p)
                st.markdown("---")
                st.markdown(f"**Expected Impact:** {issue['expected_impact']}")

    # Implementation Roadmap
    st.markdown("---")
    st.subheader("Implementation Roadmap")
    st.markdown("""
    Prioritized by impact and implementation difficulty:

    | Phase | Timeframe | Actions | Est. Score Impact |
    |-------|-----------|---------|-------------------|
    | **Phase 1: System Prompts** | 1-2 weeks | Add context-gathering, transparency, professional referral, and citation instructions to system prompts | +15-25 points |
    | **Phase 2: RAG Integration** | 2-4 weeks | Build per-dimension citation databases, professional referral templates, and resource libraries | +6-10 points |
    | **Phase 3: Fine-Tuning** | 4-8 weeks | DPO training on context-gathering, citation, and faith-aligned conversation pairs | +10-20 points |
    | **Phase 4: Benchmark Fixes** | 2-4 weeks | Recalibrate rubric weights using IRT, add per-turn scoring, expand Faith questions | Better measurement |

    **Total estimated improvement: +30-55 points** from combined interventions, potentially pushing top models past the 90-point threshold.
    """)

    # Solution-to-Rubric mapping visualization
    st.subheader("Solution Coverage Map")
    st.markdown("Which solutions address which rubric questions:")

    from data_loader import RUBRIC_QUESTIONS as RQ
    coverage_data = []
    for issue in issues:
        for q in issue["affected_questions"]:
            coverage_data.append({
                "Solution": issue["title"][:40] + "...",
                "Rubric Q": f"Q{q}",
                "Weight": RQ.get(q, {}).get("weight", 0),
                "Severity": issue["severity"],
            })

    if coverage_data:
        coverage_df = pd.DataFrame(coverage_data)
        pivot = coverage_df.pivot_table(index="Solution", columns="Rubric Q", values="Weight",
                                        aggfunc="first").fillna(0)
        fig = go.Figure(data=go.Heatmap(
            z=(pivot.values > 0).astype(int),
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#f5f5f5"], [1, "#2196F3"]],
            showscale=False,
            text=pivot.values.astype(int),
            texttemplate="%{text}",
            hovertemplate="Solution: %{y}<br>Question: %{x}<br>Weight: %{text}<extra></extra>",
        ))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=0)
        st.plotly_chart(fig, use_container_width=True)


# (separator page removed — navigation is now sectioned)


if __name__ == "__main__":
    pass
