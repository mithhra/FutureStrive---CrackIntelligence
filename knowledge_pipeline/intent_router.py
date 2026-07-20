"""
knowledge_pipeline/intent_router.py
-------------------------------------
Unified intent router for the Construction Intelligence Platform.
Routes every user query to the correct handler before any retrieval or LLM call.

Intent types:
  - "greeting"          : hi, hello, thanks, etc.
  - "analytical"        : numerical comparisons, filters, rankings on project data
  - "crack_prediction"  : explain/query the crack prediction result
  - "defect_prediction" : explain/query the defect volume prediction result
  - "knowledge"         : definitions, standards, how-to, explanations (FAISS + Qwen)
  - "off_topic"         : unrelated to construction
"""

import re


# ── Keywords for analytical / data queries ───────────────────────────────────
ANALYTICAL_KEYWORDS = (
    "less than", "greater than", "more than", "above", "below", "under",
    "top", "highest", "lowest", "average", "avg", "sum", "total",
    "maximum", "minimum", "ranking", "rank", "compare", "filter",
    "percentage", "percent", "contribution", "shap", "feature", "value",
    "show me", "list", "how many", "which parameter",
    "greater", "lesser", "exceed", "exceeds",
)

# ── Keywords that signal a CRACK prediction query ────────────────────────────
CRACK_PREDICTION_KEYWORDS = (
    "crack probability", "crack risk", "crack occur", "crack occurrence",
    "crack type", "crack severity", "crack prediction",
    "will there be a crack", "likelihood of crack", "crack flagged",
    "why is the mix flagged", "mix flagged", "w/c ratio flagged",
    "wc ratio", "water cement ratio", "honeycombing", "shrinkage crack",
    "plastic shrinkage", "remediation", "crack result",
)

# ── Keywords that signal a DEFECT VOLUME prediction query ────────────────────
DEFECT_PREDICTION_KEYWORDS = (
    "defect count", "defect volume", "defect prediction", "defect result",
    "defects per floor", "floor defects", "defect type", "defect severity",
    "defect root cause", "rework", "subcontractor quality",
    "qc compliance", "hold point", "defect rate", "defect high",
    "why is defect", "what is causing defect", "defect trend",
    "severity grade", "defect classification",
)

# ── Keywords that signal a knowledge / engineering query ─────────────────────
KNOWLEDGE_PREFIXES = (
    "what is", "what are", "who is", "tell me about", "describe",
    "explain", "how does", "how do", "how to", "how can", "how should",
    "why does", "why do", "why is", "why are", "what causes", "what happens",
    "define", "give me", "what should", "recommend", "prevent", "fix",
    "repair", "simulate", "what if", "if i", "if curing", "if w/c",
    "is 456", "cpwd", "ndt", "rebound", "upv",
    "curing", "water cement", "concrete grade",
    "safety", "formwork", "ppe", "hot weather", "inspection",
    "spi", "schedule performance", "qc hold", "third party inspection",
)

# ── Greetings ─────────────────────────────────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "greetings",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ty", "ok", "okay", "bye",
    "good", "nice", "cool", "great", "awesome", "sup", "yo",
}

# ── Broad construction domain terms ──────────────────────────────────────────
CONSTRUCTION_TERMS = (
    "crack", "concrete", "curing", "cement", "honeycombing",
    "vibration", "slump", "grade", "mix", "pour", "temperature", "humidity",
    "safety", "inspection", "ndt", "is 456", "cpwd", "boq", "delay",
    "material", "reinforcement", "rebar", "strength", "shrinkage", "flagged",
    "predict", "shap", "risk", "standard", "code", "specification",
    "rebound", "upv", "formwork", "ppe", "aggregate", "admixture",
    "defect", "rework", "subcontractor", "workforce", "qc", "hold point",
    "floor", "severity", "root cause", "spi", "compliance",
    "w/c", "wc ratio", "curing days", "placing",
)


def classify_query(query: str) -> str:
    """
    Classify the user's query into one of 6 intent types.

    Returns one of:
        "greeting"          -> respond with welcome / capability list
        "analytical"        -> pandas filter on active module's data
        "crack_prediction"  -> explain the crack prediction result
        "defect_prediction" -> explain the defect volume result
        "knowledge"         -> FAISS retrieval + Qwen LLM answer
        "off_topic"         -> politely decline
    """
    normalized = " ".join(query.lower().strip().split())

    # 1. Greeting check
    stripped = normalized.rstrip("!.?,")
    if stripped in GREETINGS or (len(normalized.split()) <= 3 and stripped in GREETINGS):
        print(f"[ROUTER] greeting <- '{query}'")
        return "greeting"

    # 2. Defect prediction intent (before generic knowledge check)
    if any(kw in normalized for kw in DEFECT_PREDICTION_KEYWORDS):
        print(f"[ROUTER] defect_prediction <- '{query}'")
        return "defect_prediction"

    # 3. Crack prediction intent
    if any(kw in normalized for kw in CRACK_PREDICTION_KEYWORDS):
        print(f"[ROUTER] crack_prediction <- '{query}'")
        return "crack_prediction"

    # 4. Analytical intent — numerical data queries
    has_analytical_kw = any(kw in normalized for kw in ANALYTICAL_KEYWORDS)
    if has_analytical_kw:
        print(f"[ROUTER] analytical <- '{query}'")
        return "analytical"

    # 5. Knowledge intent — engineering definitions, IS codes, standards
    is_construction = any(term in normalized for term in CONSTRUCTION_TERMS)
    starts_with_knowledge = any(normalized.startswith(p) for p in KNOWLEDGE_PREFIXES)
    if is_construction or starts_with_knowledge:
        print(f"[ROUTER] knowledge <- '{query}'")
        return "knowledge"

    # 6. Off-topic
    print(f"[ROUTER] off_topic <- '{query}'")
    return "off_topic"
