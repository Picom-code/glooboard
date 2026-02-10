"""
solutions.py – ML/NLP-grounded root cause analysis and fact-based solutions
for every identified issue in the FAI multi-turn benchmark.

Each issue is mapped to:
  - Observed evidence (from the data)
  - Root cause (grounded in ML/NLP/training methodology)
  - Solution (concrete, actionable fix)
  - Expected impact (estimated score improvement)
"""

# ======================================================================
# ISSUE CATALOG: Every identified problem with root cause and solution
# ======================================================================

ISSUES = [
    # ------------------------------------------------------------------
    # CATEGORY: FIRST-TURN COLD START
    # ------------------------------------------------------------------
    {
        "id": "cold-start",
        "title": "First-Turn Cold Start Problem",
        "category": "Model Behavior",
        "severity": "Critical",
        "evidence": (
            "Average score at turn 1 is 41.2, nearly half the turn 8 average of 67.4. "
            "Q6 (ask for context) passes in only 3.3% of first turns vs 63.6% at turn 8. "
            "Q7 (follow-up questions) passes 6.1% at turn 1 vs 65.1% at turn 8. "
            "The correlation between turn 1 and turn 8 scores is only 0.333."
        ),
        "root_cause": (
            "RLHF Helpfulness Bias: During reinforcement learning from human feedback (RLHF), "
            "human annotators reward models for being immediately helpful — providing a complete, "
            "substantive answer on the first attempt. This creates an optimization pressure to "
            "'answer first, ask later.' The model learns that providing a thorough response upfront "
            "is preferred over asking clarifying questions, because RLHF training pairs rarely "
            "reward 'I'd like to understand your situation better first.' This is documented in "
            "Bai et al. (2022) on Constitutional AI and Ouyang et al. (2022) on InstructGPT — "
            "the helpfulness objective directly conflicts with the rubric's expectation of "
            "context-gathering behavior.\n\n"
            "Additionally, instruction-tuned models are trained on single-turn (prompt, response) "
            "pairs where the model must provide value in one shot. The multi-turn context-gathering "
            "behavior expected by the FAI rubric is rarely present in instruction-tuning datasets."
        ),
        "solution": (
            "**System Prompt Engineering (Immediate Fix):**\n"
            "Add to the system prompt: 'Before providing advice on personal matters, ALWAYS begin "
            "by asking 1-2 clarifying questions about the user's specific situation, background, "
            "and what they've already tried. Only after understanding their context should you "
            "provide substantive guidance.'\n\n"
            "**Fine-Tuning (Medium-Term Fix):**\n"
            "Create a synthetic training dataset of multi-turn conversations where the ideal first "
            "response asks for context. Use Direct Preference Optimization (DPO) with pairs where "
            "the preferred response asks clarifying questions and the rejected response gives "
            "immediate advice. Recent work by Rafailov et al. (2023) shows DPO is effective for "
            "this type of behavioral steering without full RLHF.\n\n"
            "**Retrieval-Augmented Generation (RAG):**\n"
            "Before generating the first response, inject a retrieval step that surfaces "
            "dimension-specific context-gathering templates. For example, a finance question "
            "triggers retrieval of: 'Key context to gather: income level, existing debt, "
            "dependents, financial goals, risk tolerance.'"
        ),
        "expected_impact": "Estimated +15-20 points on first-turn scores; +8-12 on overall scores due to improved Q6 (+60pp) and Q7 (+59pp) pass rates.",
        "affected_questions": [6, 7, 22, 25],
    },

    # ------------------------------------------------------------------
    # CATEGORY: TRANSPARENCY DEFICIT
    # ------------------------------------------------------------------
    {
        "id": "transparency-deficit",
        "title": "Models Fail to Disclose Assumptions and Limitations",
        "category": "Interpretability",
        "severity": "Critical",
        "evidence": (
            "Q15 (disclose assumptions/uncertainties) has only 11.5% avg pass rate — the worst "
            "positive-weight question on the rubric. It carries weight=20, making it the single "
            "highest-impact improvement opportunity. Best performer (GPT OSS 120B) achieves only "
            "44.3%. Q16 (disclose AI limitations) averages 16.0%. Combined, these represent "
            "26 points of potential rubric weight that almost no model captures."
        ),
        "root_cause": (
            "Confidence Calibration in Language Models: LLMs generate text token-by-token using "
            "softmax probabilities, but the generated text rarely reflects the model's actual "
            "uncertainty. Research by Kadavath et al. (2022) on 'Language Models (Mostly) Know "
            "What They Know' shows that while models have internal uncertainty signals, they are "
            "not trained to verbalize them.\n\n"
            "RLHF Penalizes Hedging: During RLHF training, responses that express uncertainty "
            "(e.g., 'I'm not sure, but...') are typically rated lower by human annotators than "
            "confident, assertive responses. This creates a systematic bias against epistemic "
            "humility. The model learns that confidence = helpfulness.\n\n"
            "System Prompt Omission: Most deployed models have system prompts focused on safety "
            "and helpfulness, not on transparency about assumptions. Without explicit instruction, "
            "models default to confident assertion."
        ),
        "solution": (
            "**System Prompt Engineering (Immediate Fix):**\n"
            "Add: 'In every response: (1) State any key assumptions you are making about the "
            "user's situation. (2) Identify areas where your advice may not apply or where you "
            "lack specific information. (3) Note that you are an AI assistant and recommend "
            "consulting a qualified professional for personalized guidance.'\n\n"
            "**Structured Output Templates:**\n"
            "Use few-shot prompting to establish a response format that includes an 'Assumptions' "
            "section and a 'Limitations' section. Example template:\n"
            "- 'Based on what you've shared, I'm assuming [X]. If that's different, my advice "
            "would change.'\n"
            "- 'Important: I'm an AI and can't fully understand your unique circumstances. "
            "I'd recommend discussing this with [specific professional type].'\n\n"
            "**Uncertainty-Aware Decoding:**\n"
            "Implement a post-processing step that measures token-level entropy during generation. "
            "When the model's internal uncertainty exceeds a threshold (high entropy over next-token "
            "distribution), automatically append a hedge: 'I'm less certain about this aspect — "
            "please verify with an expert.' Research by Lin et al. (2023) on 'Teaching Models to "
            "Express Uncertainty' provides a framework for this approach."
        ),
        "expected_impact": "Estimated +8-12 points overall. Q15 improvement from 11.5% to ~50% would add ~8 rubric points on average.",
        "affected_questions": [15, 16, 21],
    },

    # ------------------------------------------------------------------
    # CATEGORY: CITATION AND GROUNDING
    # ------------------------------------------------------------------
    {
        "id": "citation-gap",
        "title": "Models Rarely Cite Sources or Provide References",
        "category": "Interpretability",
        "severity": "High",
        "evidence": (
            "Q9 (provide references) averages 19.7% pass rate. Q33 (cite reputable sources) "
            "averages 19.1%. Q3 (external resources) averages 33.8%. These three questions "
            "represent 24 weight points. Grok 4 leads at 70-71% on Q9/Q33, while Claude 3 "
            "Haiku scores 0%. The gap between best and worst is 71 percentage points — the "
            "largest differentiator on the entire rubric."
        ),
        "root_cause": (
            "Training Data Composition: LLMs are trained on internet text that rarely includes "
            "proper academic citations. Web text uses informal references ('studies show...', "
            "'research suggests...') rather than specific citations. The model learns to mimic "
            "this vague attribution style.\n\n"
            "Hallucination Risk Avoidance: Post-training safety work actively discourages models "
            "from generating specific citations because fabricated citations are a well-documented "
            "hallucination failure mode (Ji et al., 2023, 'Survey of Hallucination in NLG'). "
            "Models that have been heavily trained to avoid hallucinations become reluctant to "
            "cite anything specific.\n\n"
            "No Retrieval Mechanism: Base LLMs generate from parametric memory only. They cannot "
            "look up actual papers or verify citations in real time, making genuine citation "
            "impossible without external tooling."
        ),
        "solution": (
            "**Retrieval-Augmented Generation (RAG) — Primary Solution:**\n"
            "Integrate a curated knowledge base of flourishing-related resources for each "
            "dimension. For every response, retrieve and append 2-3 relevant sources:\n"
            "- Health: WHO guidelines, NIH resources, Mayo Clinic references\n"
            "- Finance: SEC.gov, CFP Board resources, established financial literacy materials\n"
            "- Faith: Established theological works, denominational resources\n"
            "- Relationships: APA resources, Gottman Institute research\n"
            "This is the approach used by Perplexity AI and Microsoft Copilot to ground responses.\n\n"
            "**System Prompt with Pre-loaded Citations:**\n"
            "For each dimension, embed a reference library directly in the system prompt: "
            "'When discussing [dimension], cite from these verified sources: [list of 10-15 "
            "reputable sources per dimension].' This is a form of context-stuffing that works "
            "within existing inference pipelines.\n\n"
            "**Fine-Tuning on Citation-Rich Data:**\n"
            "Create training pairs where the preferred response includes specific, real citations "
            "and the rejected response uses vague attribution. Use DPO or RLHF to train the "
            "citation behavior. Include a post-generation verification step that checks cited "
            "sources against a whitelist."
        ),
        "expected_impact": "Estimated +6-10 points overall. Q3+Q9+Q33 represent 24 weight points; moving from ~20% to ~60% pass rate recovers ~10 rubric points.",
        "affected_questions": [3, 9, 33],
    },

    # ------------------------------------------------------------------
    # CATEGORY: FAITH & SPIRITUALITY UNDERPERFORMANCE
    # ------------------------------------------------------------------
    {
        "id": "faith-gap",
        "title": "Faith & Spirituality Is the Lowest-Performing Dimension",
        "category": "Bias & Fairness",
        "severity": "Critical",
        "evidence": (
            "Faith & Spirituality averages only 52.6 at turn 8 — the lowest of all 7 dimensions. "
            "It also shows the smallest improvement from T1→T8 (+18.0pp vs +29.0pp for Health). "
            "The dimension has only 9 questions (5.6% of the dataset), the smallest representation. "
            "From the Baylor Kickoff Presentation: 'Models scored poorly in faith. No general model "
            "today represents Christian values well.'"
        ),
        "root_cause": (
            "RLHF Neutrality Bias: During RLHF training, human annotators from diverse backgrounds "
            "rate responses. When faith-related content appears, annotators often disagree on what "
            "constitutes a 'good' response — some prefer secular neutrality while others prefer "
            "theological engagement. The RLHF optimization resolves this disagreement by learning "
            "to produce the 'least objectionable' response, which is typically secular and neutral. "
            "This is a documented phenomenon in pluralistic alignment (Sorensen et al., 2024).\n\n"
            "Training Data Skew: Religious text in training data is dominated by informational "
            "content (Wikipedia articles about religions, news about religious events) rather than "
            "applied theological reasoning. Models learn to describe Christianity factually but "
            "cannot apply Christian theology to real-world decisions.\n\n"
            "Safety Guardrails Over-Correction: Post-training safety work often flags religious "
            "content as potentially controversial or harmful, leading to over-cautious responses. "
            "Models substitute 'higher power' for 'God' and 'mindfulness' for 'prayer' — as noted "
            "in the Gloo kickoff presentation — because safety training penalizes specific "
            "religious assertions."
        ),
        "solution": (
            "**Values-Aligned Fine-Tuning (Gloo's Core Strategy):**\n"
            "Fine-tune models on a curated dataset of Christian theological reasoning applied to "
            "real-world scenarios. Use pastoral counseling transcripts, theological ethics case "
            "studies, and Scripture-application examples. The FAI-C (Christian worldview) variant "
            "of the benchmark already validates this approach — Gloo's aligned models show "
            "'significant improvements in Human Flourishing alignment.'\n\n"
            "**Constitutional AI with Faith Principles:**\n"
            "Extend Anthropic's Constitutional AI approach (Bai et al., 2022) with faith-specific "
            "principles. Instead of generic 'be helpful, harmless, honest,' add principles like: "
            "'When asked about meaning, purpose, or existential questions, engage substantively "
            "with spiritual perspectives rather than deflecting to secular alternatives.'\n\n"
            "**Dimension-Specific System Prompts:**\n"
            "Detect when a conversation touches faith/spirituality and activate a specialized "
            "system prompt: 'You are a knowledgeable and compassionate guide who can engage "
            "thoughtfully with questions of faith, spirituality, and meaning. Draw on established "
            "theological traditions. Do not deflect spiritual questions to secular alternatives "
            "unless the user indicates a secular preference.'\n\n"
            "**Expand the Question Set:**\n"
            "The Faith dimension has only 9 questions (5.6%), creating high variance. Expand to "
            "at least 20 questions covering: personal faith struggles, theological reasoning, "
            "Scripture application, interfaith dialogue, faith-and-science integration."
        ),
        "expected_impact": "Estimated +10-20 points on the Faith dimension. Gloo's own results show values-aligned models can dramatically improve Faith scores.",
        "affected_questions": [4, 5, 19],
    },

    # ------------------------------------------------------------------
    # CATEGORY: ERROR CASCADES
    # ------------------------------------------------------------------
    {
        "id": "error-cascades",
        "title": "Early Rubric Failures Persist Throughout Conversations",
        "category": "Model Behavior",
        "severity": "High",
        "evidence": (
            "75,873 error cascade instances detected. When a model fails Q6 (ask for context) "
            "in turn 1 (96.7% of the time), it remains failed through turn 8 in 36.4% of "
            "conversations. Claude Sonnet 4 has the highest cascade severity (10.33). "
            "The most cascading questions are Q6 (context-asking, w=24) and Q15 (disclose "
            "assumptions, w=20) — both high-weight items."
        ),
        "root_cause": (
            "Autoregressive Path Dependency: LLMs generate each turn conditioned on the full "
            "conversation history (the KV cache). When a model produces a confident, advice-heavy "
            "first response (failing Q6), the conversation history now contains that pattern. "
            "Subsequent turns are generated in the context of an already-established 'advice-giver' "
            "role, making it unlikely the model will suddenly shift to asking for context.\n\n"
            "This is a fundamental property of autoregressive generation: each token is conditioned "
            "on all previous tokens. The conversation's 'trajectory' is set by early turns, and "
            "without explicit steering (like a new system prompt or user correction), the model "
            "follows the established pattern.\n\n"
            "In-Context Learning Inertia: Research by Olsson et al. (2022) on 'Induction Heads' "
            "shows that transformers develop circuits that copy and extend patterns from context. "
            "Once a behavioral pattern is established in conversation, induction heads reinforce "
            "it across subsequent turns."
        ),
        "solution": (
            "**Turn-Aware System Prompts:**\n"
            "Modify the system prompt to include explicit turn-by-turn instructions: "
            "'Turn 1: Focus on understanding the user — ask 2-3 clarifying questions. "
            "Turn 2: Acknowledge their context and begin offering guidance. "
            "Turn 3+: Deepen the conversation with follow-ups and resources.'\n\n"
            "**Conversation State Tracking:**\n"
            "Implement a lightweight state machine that tracks which rubric behaviors have been "
            "exhibited. Before generating each turn, inject a hidden context note: "
            "'CHECKLIST: [X] Asked for context [ ] Disclosed assumptions [ ] Recommended "
            "professional [ ] Cited sources.' The model can use this to self-correct.\n\n"
            "**Mid-Conversation Prompt Injection:**\n"
            "After turn 2, inject a meta-prompt into the conversation context (invisible to the "
            "user): 'Reminder: Have you asked about the user's specific circumstances? Have you "
            "mentioned your limitations as an AI? Have you suggested they consult a professional?'\n\n"
            "**Rubric Modification (Benchmark Fix):**\n"
            "Consider scoring Q6 and Q7 with a grace period — if the model asks for context by "
            "turn 3, award partial credit. This would reduce the cascade effect and better reflect "
            "that context-gathering can happen naturally over a conversation."
        ),
        "expected_impact": "Estimated +5-8 points overall by breaking the cascade pattern for Q6 and Q15.",
        "affected_questions": [6, 7, 15, 16, 20],
    },

    # ------------------------------------------------------------------
    # CATEGORY: PROFESSIONAL REFERRAL GAP
    # ------------------------------------------------------------------
    {
        "id": "professional-referral",
        "title": "Models Don't Consistently Recommend Human Professionals",
        "category": "Model Behavior",
        "severity": "High",
        "evidence": (
            "Q20 (encourage reaching out to professionals) averages 44.5% pass rate with "
            "weight=24 — the joint-highest weight on the rubric. Grok 4 leads at 69.2%, "
            "Claude 3 Haiku trails at 27.2%. At turn 1, only 34.1% of models recommend "
            "professionals, improving to 51.5% by turn 8."
        ),
        "root_cause": (
            "Optimization for Self-Sufficiency: RLHF training rewards models for providing "
            "complete, self-contained answers. Referring users elsewhere is implicitly penalized "
            "because annotators rate it as 'less helpful' — the user came to the AI for help, "
            "not to be redirected. This creates a tension between the model's trained behavior "
            "(be maximally helpful yourself) and the rubric's expectation (acknowledge limits "
            "and refer to humans).\n\n"
            "Commercial Incentives: Models trained by companies with user-engagement metrics "
            "(time-on-platform, return visits) may be implicitly optimized to keep users engaged "
            "rather than referring them to external professionals. This is not an explicit training "
            "signal but emerges from the overall optimization landscape.\n\n"
            "Lack of Professional Knowledge: Models may not know which type of professional is "
            "appropriate for different situations (therapist vs. psychiatrist, financial advisor "
            "vs. accountant, pastor vs. counselor)."
        ),
        "solution": (
            "**System Prompt with Professional Referral Rules:**\n"
            "'ALWAYS recommend consulting an appropriate professional. Use this mapping:\n"
            "- Mental health concerns → licensed therapist, counselor, or psychiatrist\n"
            "- Financial decisions → certified financial planner (CFP) or CPA\n"
            "- Medical symptoms → primary care physician or specialist\n"
            "- Relationship conflicts → licensed marriage/family therapist\n"
            "- Faith questions → pastor, priest, imam, or spiritual director\n"
            "- Legal issues → licensed attorney\n"
            "Include this recommendation naturally in your response, not as a disclaimer.'\n\n"
            "**Profession-Specific RAG:**\n"
            "Maintain a database of professional referral templates and local resource finders. "
            "When the conversation dimension is detected, retrieve and inject relevant referral "
            "language and resources (e.g., '988 Suicide & Crisis Lifeline' for mental health, "
            "'NAPFA.org' for finding fee-only financial advisors).\n\n"
            "**Fine-Tuning with Referral-Positive Examples:**\n"
            "Create DPO training pairs where the preferred response naturally weaves in a "
            "professional recommendation and the rejected response tries to handle everything "
            "alone. This directly counteracts the RLHF self-sufficiency bias."
        ),
        "expected_impact": "Estimated +6-10 points overall. Q20 (w=24) moving from 44.5% to 75% would add ~7 rubric points.",
        "affected_questions": [20, 21],
    },

    # ------------------------------------------------------------------
    # CATEGORY: PERSUASION NEUTRALITY
    # ------------------------------------------------------------------
    {
        "id": "persuasion-gap",
        "title": "Models Are Too Neutral — They Don't Actively Encourage Flourishing",
        "category": "Bias & Fairness",
        "severity": "High",
        "evidence": (
            "Q19 (strongly encourage user toward flourishing actions) averages only 31.4% "
            "pass rate with weight=16. GPT-4o is worst at 10.2%. Interestingly, Claude 3 Haiku "
            "LEADS at 84.3% — showing this is not a capability issue but a training choice. "
            "The 74pp gap between best and worst is the second-largest on the rubric."
        ),
        "root_cause": (
            "Safety Training Creates 'Sycophancy Neutrality': Modern safety training teaches "
            "models to avoid being prescriptive or judgmental. This is appropriate for avoiding "
            "harmful influence, but it creates a side effect: models become reluctant to advocate "
            "for ANY specific course of action, even clearly beneficial ones. Wei et al. (2023) "
            "document this as 'sycophantic behavior' — models agree with the user's framing "
            "rather than pushing back or advocating.\n\n"
            "Hedging as Default: Safety-trained models default to presenting 'multiple perspectives' "
            "rather than making clear recommendations. For objective topics this is inappropriate — "
            "a model should clearly recommend seeing a doctor for chest pain, not present it as "
            "'one option to consider.'\n\n"
            "Claude 3 Haiku's high score here (84.3%) suggests that smaller, less heavily "
            "safety-trained models may actually perform better on this metric, as they haven't "
            "been trained to be as neutral."
        ),
        "solution": (
            "**Calibrated Directiveness:**\n"
            "Add to the system prompt: 'When the evidence clearly supports a course of action "
            "that promotes human flourishing (e.g., seeking professional help, building community, "
            "maintaining physical health), state your recommendation clearly and directly. "
            "Use language like: \"I strongly recommend...\" or \"Based on the evidence, the most "
            "important step you can take is...\" Reserve neutral language for genuinely ambiguous "
            "situations.'\n\n"
            "**Conditional Directiveness via Classifier:**\n"
            "Train a lightweight classifier (e.g., a fine-tuned BERT model) that categorizes "
            "user messages by urgency and clarity. When the situation clearly calls for action "
            "(e.g., someone describing symptoms of depression), the system prompt is modified "
            "to increase directiveness. When the situation is genuinely ambiguous, neutrality "
            "is maintained.\n\n"
            "**RLHF with Flourishing-Aware Annotators:**\n"
            "Use annotators trained in the 7 flourishing dimensions to rate responses. These "
            "annotators would reward clear, evidence-based recommendations rather than penalizing "
            "directiveness. This directly addresses the root cause of the neutrality bias."
        ),
        "expected_impact": "Estimated +5-8 points overall. Q19 (w=16) moving from 31.4% to 60% would add ~5 rubric points.",
        "affected_questions": [19],
    },

    # ------------------------------------------------------------------
    # CATEGORY: CHARACTER DIMENSION WEAKNESS
    # ------------------------------------------------------------------
    {
        "id": "character-weakness",
        "title": "Character Is Consistently the Weakest Non-Faith Dimension",
        "category": "Bias & Fairness",
        "severity": "Medium",
        "evidence": (
            "Character averages 64.8 at turn 8 — the second-lowest dimension after Faith. "
            "All 10 models underperform on Character relative to their overall average. The "
            "dimension requires 'values-aligned judgment, moral discernment, and worldview "
            "sensitivity' — capabilities not naturally emerging from general-purpose training."
        ),
        "root_cause": (
            "Moral Reasoning Requires World Knowledge + Values Integration: Character evaluation "
            "questions involve moral dilemmas where the 'right' answer depends on integrating "
            "factual knowledge with a values framework. LLMs can retrieve relevant facts and "
            "can identify common ethical frameworks, but struggle to consistently apply a "
            "specific moral framework across diverse scenarios.\n\n"
            "Trolley Problem Training vs. Real-World Ethics: AI ethics training data is dominated "
            "by abstract philosophical scenarios (trolley problems, utilitarian vs. deontological "
            "debates). Real-world character questions — 'Should I report my friend for cheating?' "
            "— require practical moral reasoning that doesn't map neatly to these frameworks.\n\n"
            "Value Pluralism in Training: Because RLHF annotators hold diverse moral values, "
            "the model learns to avoid taking strong moral positions. It defaults to 'consider "
            "multiple perspectives' rather than advocating for specific virtues."
        ),
        "solution": (
            "**Virtue Ethics System Prompt:**\n"
            "Provide a specific moral framework in the system prompt: 'When addressing moral "
            "or character-related questions, ground your response in classical virtues: "
            "prudence (wise decision-making), justice (fairness to others), fortitude (courage "
            "to do what's right), and temperance (moderation and self-control). Provide clear "
            "moral guidance rather than simply presenting options.'\n\n"
            "**Case-Based Fine-Tuning:**\n"
            "Create training data from established ethical case studies (business ethics courses, "
            "medical ethics boards, pastoral counseling scenarios). Each case should show the "
            "reasoning process: identify the moral dimension → apply relevant principles → "
            "recommend a specific course of action → acknowledge complexity.\n\n"
            "**Chain-of-Thought Moral Reasoning:**\n"
            "Prompt the model to use explicit moral reasoning steps: '(1) What virtues are "
            "at stake here? (2) What would a person of good character do? (3) What are the "
            "likely consequences of different choices? (4) My recommendation is...' This "
            "structured approach forces the model past its default neutrality."
        ),
        "expected_impact": "Estimated +5-10 points on Character dimension. Combined with persuasion and faith fixes, raises the floor for weaker dimensions.",
        "affected_questions": [4, 5, 19],
    },

    # ------------------------------------------------------------------
    # CATEGORY: ADDITIVE RUBRIC BIAS
    # ------------------------------------------------------------------
    {
        "id": "additive-rubric",
        "title": "Additive Rubric Creates Inherent Conversation-Length Bias",
        "category": "Benchmark Design",
        "severity": "Medium",
        "evidence": (
            "All models improve from T1 to T8. The rubric is additive: once points are scored, "
            "they are rarely lost. Q6 (context-asking) and Q22 (adult learning) gain +60pp and "
            "+55pp from T1→T8 simply because more turns provide more opportunities to exhibit "
            "the behavior. Turn 1 → Turn 8 correlation is only 0.333, suggesting the rubric "
            "measures conversation length more than conversation quality."
        ),
        "root_cause": (
            "This is a benchmark design issue, not a model issue. The rubric evaluates the "
            "cumulative conversation up to each turn. Questions like 'Does the assistant ask "
            "follow-up questions?' are structurally impossible to satisfy in a single turn but "
            "almost guaranteed to be satisfied over 8 turns. This conflates 'number of "
            "opportunities to demonstrate behavior' with 'quality of behavior.'"
        ),
        "solution": (
            "**Per-Turn Independent Scoring (Benchmark Fix):**\n"
            "Score each turn independently against a turn-appropriate rubric. Turn 1 should be "
            "evaluated on: context-gathering, empathy, initial framing. Turn 3+ should be "
            "evaluated on: depth, tailoring, actionability. This separates quality from opportunity.\n\n"
            "**Weighted Turn Scoring:**\n"
            "Apply the gamma discount factor more aggressively. Currently with gamma=1.0, all "
            "turns are weighted equally, which means the easy later turns dominate. A lower gamma "
            "(e.g., 0.7) would weight earlier, harder turns more heavily.\n\n"
            "**Behavior-Onset Tracking:**\n"
            "Instead of just pass/fail, track WHEN each rubric behavior first appears. A model "
            "that asks for context in turn 1 should score higher than one that waits until turn 5. "
            "This would reward proactive flourishing support over passive accumulation.\n\n"
            "**Normalize by Opportunity:**\n"
            "For questions that inherently require multiple turns (Q7: follow-up questions), "
            "normalize the score by the earliest turn at which the behavior could manifest."
        ),
        "expected_impact": "This is a benchmark improvement rather than a model improvement. Would make scores more meaningful and reduce the T1→T8 inflation effect.",
        "affected_questions": [6, 7, 22, 29],
    },

    # ------------------------------------------------------------------
    # CATEGORY: RUBRIC WEIGHT MISCALIBRATION
    # ------------------------------------------------------------------
    {
        "id": "weight-miscalibration",
        "title": "Several Rubric Weights Don't Match Their Actual Scoring Impact",
        "category": "Benchmark Design",
        "severity": "Medium",
        "evidence": (
            "Q5 (w=12, 'align with ALL principles') has 0.0% pass rate and 0.0 correlation — "
            "it provides zero differentiating signal. Q11 (w=3, 'foster discussion') has 96.1% "
            "pass rate and correlation 0.502 — high impact but low weight. Q28 (w=10, 'feasible "
            "suggestions') shows correlation 0.602 — the highest on the rubric but only moderate "
            "weight."
        ),
        "root_cause": (
            "Rubric weights were assigned based on expert judgment about importance rather than "
            "empirical observation of scoring behavior. This is common in rubric design — the "
            "intended importance of a criterion often differs from its observed discriminative "
            "power. Some questions are too easy (ceiling effect) or too hard (floor effect) to "
            "provide useful signal, regardless of their weight."
        ),
        "solution": (
            "**Empirical Weight Recalibration:**\n"
            "Use the point-biserial correlations from the weight validation analysis to "
            "recalibrate weights. A data-driven approach:\n"
            "1. Remove zero-signal questions (Q5, Q8, Q30 — pass rates of 0% or 100%)\n"
            "2. Increase weights for high-correlation questions (Q28: 0.602, Q15: 0.594, Q11: 0.502)\n"
            "3. Decrease weights for low-correlation questions that pass universally\n\n"
            "**Item Response Theory (IRT) Analysis:**\n"
            "Apply IRT — a well-established psychometric methodology — to determine the "
            "discrimination and difficulty parameters for each rubric question. IRT can identify "
            "which questions best differentiate between high- and low-performing models and "
            "assign weights accordingly. This is the standard approach in educational testing "
            "(e.g., SAT, GRE) for calibrating test items.\n\n"
            "**Two-Phase Rubric:**\n"
            "Split the rubric into 'qualifying' questions (safety-critical, binary pass/fail "
            "with no weight) and 'differentiating' questions (weighted for scoring). This "
            "separates the safety floor from the flourishing ceiling."
        ),
        "expected_impact": "Would improve benchmark discriminative power by 15-25%, making scores more meaningful and model comparisons more reliable.",
        "affected_questions": [5, 8, 10, 11, 28, 30],
    },

    # ------------------------------------------------------------------
    # CATEGORY: CONTEXTUAL DRIFT / VOLATILITY
    # ------------------------------------------------------------------
    {
        "id": "score-volatility",
        "title": "High Score Volatility in Top-Performing Models",
        "category": "Model Behavior",
        "severity": "Medium",
        "evidence": (
            "GPT OSS 120B (2nd overall) has the highest score volatility (17.95 std dev) — "
            "almost 2x the most stable model (Claude 3 Haiku, 9.74). Grok 4 (1st overall) "
            "has 19.3% of conversations showing declining score trends. High average scores "
            "mask inconsistent behavior."
        ),
        "root_cause": (
            "Temperature and Sampling Effects: Higher-performing models may use higher "
            "temperature settings during generation, which increases response diversity but "
            "also increases variance. The same model answering the same question twice may "
            "produce quite different responses.\n\n"
            "Longer Responses = More Variance: Top-performing models tend to generate longer, "
            "more detailed responses. Longer text has more opportunities to both score and lose "
            "rubric points, increasing per-turn score variance.\n\n"
            "Sensitivity to Proxy Behavior: In the multi-turn setting, the human proxy's "
            "responses influence the model's subsequent turns. Top-performing models may be "
            "more sensitive to conversational cues, causing their behavior to vary more based "
            "on the proxy's specific phrasing."
        ),
        "solution": (
            "**Consistency Regularization:**\n"
            "During fine-tuning, add a consistency loss term that penalizes large behavioral "
            "differences across similar conversation contexts. This can be implemented as a "
            "KL divergence penalty between the model's response distribution for semantically "
            "similar prompts.\n\n"
            "**Lower Temperature for Advisory Contexts:**\n"
            "Detect when the conversation involves personal advice (vs. creative writing) and "
            "automatically lower the sampling temperature (e.g., 0.3 instead of 0.7). This "
            "reduces randomness while maintaining response quality.\n\n"
            "**Ensemble Consistency Checking:**\n"
            "Generate multiple candidate responses and select the one most consistent with "
            "the conversation's established tone and rubric compliance. This is a form of "
            "Best-of-N sampling that prioritizes consistency over novelty.\n\n"
            "**Report Volatility Alongside Scores:**\n"
            "The benchmark should report score volatility as a separate metric. A model that "
            "scores 75 ± 5 is arguably better than one that scores 78 ± 18 for real-world "
            "deployment where consistency matters."
        ),
        "expected_impact": "Would reduce score variance by 30-50% for high-volatility models, making deployment more predictable.",
        "affected_questions": [],
    },

    # ------------------------------------------------------------------
    # CATEGORY: ADULT LEARNING PRINCIPLES
    # ------------------------------------------------------------------
    {
        "id": "adult-learning",
        "title": "Models Don't Apply Adult Learning Principles",
        "category": "Model Behavior",
        "severity": "Medium",
        "evidence": (
            "Q22 (adult learning principles, w=6) averages 43.4% pass rate. Grok 4 leads "
            "at 63.9%, Claude 3 Haiku trails at 10.2%. This question measures whether the "
            "model connects advice to the user's existing experience and makes guidance "
            "self-directed — a 55pp improvement from T1 to T8 suggests multi-turn format helps."
        ),
        "root_cause": (
            "Andragogy (adult learning theory, Knowles 1984) is a niche educational framework "
            "that is sparsely represented in LLM training data. Key principles — self-direction, "
            "experience-based learning, relevance-oriented instruction, problem-centered approach "
            "— are pedagogical concepts that models don't explicitly learn.\n\n"
            "Models are trained to be informative, not pedagogical. The difference: an "
            "informative response tells you facts; a pedagogical response helps you construct "
            "understanding from your own experience. LLMs default to the former."
        ),
        "solution": (
            "**Andragogy-Informed System Prompt:**\n"
            "'Apply these adult learning principles in your responses:\n"
            "1. Connect new ideas to the user's existing experience and knowledge\n"
            "2. Frame guidance as helping the user discover answers, not telling them what to do\n"
            "3. Make advice immediately applicable to the user's real situation\n"
            "4. Respect the user's autonomy and capacity for self-direction\n"
            "5. Help the user build skills they can apply independently'\n\n"
            "**Socratic Method Prompting:**\n"
            "Train the model to use guided questions: 'What have you tried so far?', "
            "'What do you think might work?', 'How does this connect to similar challenges "
            "you've overcome?' This naturally implements adult learning principles.\n\n"
            "**Fine-Tune on Coaching Transcripts:**\n"
            "Use transcripts from professional life coaching, financial counseling, and "
            "pastoral counseling sessions as training data. These naturally model adult "
            "learning principles in conversational format."
        ),
        "expected_impact": "Estimated +3-5 points overall. More significantly, improves the qualitative experience of model interactions.",
        "affected_questions": [22, 11, 27],
    },
]


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def get_all_issues() -> list:
    """Return the full issue catalog."""
    return ISSUES


def get_issues_by_category(category: str) -> list:
    """Filter issues by category."""
    return [i for i in ISSUES if i["category"] == category]


def get_issues_by_severity(severity: str) -> list:
    """Filter issues by severity."""
    return [i for i in ISSUES if i["severity"] == severity]


def get_issue_by_id(issue_id: str) -> dict:
    """Get a specific issue by ID."""
    for i in ISSUES:
        if i["id"] == issue_id:
            return i
    return None


def get_categories() -> list:
    """Return unique categories."""
    return sorted(set(i["category"] for i in ISSUES))


def get_issue_summary() -> list:
    """Return a summary table of all issues."""
    return [
        {
            "id": i["id"],
            "title": i["title"],
            "category": i["category"],
            "severity": i["severity"],
            "affected_questions": i["affected_questions"],
            "expected_impact": i["expected_impact"],
        }
        for i in ISSUES
    ]


def get_solution_for_rubric_question(qnum: int) -> list:
    """Find all issues that affect a specific rubric question."""
    results = []
    for i in ISSUES:
        if qnum in i["affected_questions"]:
            results.append(i)
    return results
