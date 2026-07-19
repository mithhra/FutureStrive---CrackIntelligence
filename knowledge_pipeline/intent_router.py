"""
knowledge_pipeline/intent_router.py
-------------------------------------
Routes every user query to the correct handler before any retrieval or LLM call.

Intent types:
  - "greeting"    : hi, hello, thanks, etc.
  - "analytical"  : numerical comparisons, filters, rankings on project data
  - "prediction"  : run/explain crack prediction, probability, risk score
  - "knowledge"   : definitions, standards, how-to, explanations (FAISS + Qwen)
  - "off_topic"   : unrelated to construction
"""

import re


# ── Keywords that signal an analytical / data query ─────────────────────────
ANALYTICAL_KEYWORDS = (
    "less than", "greater than", "more than", "above", "below", "under",
    "top", "highest", "lowest", "average", "avg", "count", "sum", "total",
    "maximum", "minimum", "ranking", "rank", "compare", "filter",
    "percentage", "percent", "contribution", "shap", "feature", "value",
    "show me", "list", "how many", "which parameter", "what is the",
    "greater", "lesser", "exceed", "exceeds",
)

# ── Keywords that signal a prediction / risk query ────────────────────────────
PREDICTION_KEYWORDS = (
    "predict", "prediction", "crack probability", "risk score", "crack risk",
    "will there be a crack", "crack occur", "likelihood", "probability",
    "flagged", "why is the mix flagged", "what is the risk",
    "run prediction", "crack occurrence", "crack type", "crack severity",
    "remediation cost",
)

# ── Keywords that signal a knowledge / engineering query ──────────────────────
KNOWLEDGE_PREFIXES = (
    "what is", "what are", "who is", "tell me about", "describe",
    "explain", "how does", "how do", "how to", "how can", "how should",
    "why does", "why do", "why is", "why are", "what causes", "what happens",
    "define", "give me", "what should", "recommend", "prevent", "fix",
    "repair", "simulate", "what if", "if i", "if curing", "if w/c",
    "is 456", "cpwd", "ndt", "rebound", "upv", "honeycombing",
    "curing", "water cement", "w/c", "concrete grade", "shrinkage",
    "safety", "formwork", "ppe", "hot weather", "inspection",
)

# ── Greetings ─────────────────────────────────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "greetings",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ty", "ok", "okay", "bye",
    "good", "nice", "cool", "great", "awesome", "sup", "yo",
}

# ── Construction domain check ─────────────────────────────────────────────────
CONSTRUCTION_TERMS = (
    "crack", "concrete", "curing", "water", "cement", "honeycombing",
    "vibration", "slump", "grade", "mix", "pour", "temperature", "humidity",
    "safety", "inspection", "ndt", "is 456", "cpwd", "boq", "cost", "delay",
    "material", "reinforcement", "rebar", "strength", "shrinkage", "flagged",
    "predict", "shap", "risk", "standard", "code", "specification",
    "rebound", "upv", "formwork", "ppe", "aggregate", "admixture", "steel",
    "less than", "greater than", "contribution", "feature", "parameter",
    "w/c", "wc ratio", "curing days", "placing", "remediation",
)


def classify_query(query: str) -> str:
    """
    Classify the user's query into one of 5 intent types.

    Returns one of:
        "greeting"    → respond with welcome/capability list
        "analytical"  → run pandas filter/aggregation on project data
        "prediction"  → run or explain the crack prediction model
        "knowledge"   → FAISS retrieval + Qwen LLM answer
        "off_topic"   → politely decline
    """
    normalized = " ".join(query.lower().strip().split())

    # 1. Greeting check
    stripped = normalized.rstrip("!.?,")
    if stripped in GREETINGS or (len(normalized.split()) <= 3 and stripped in GREETINGS):
        print(f"[ROUTER] greeting  <- '{query}'")
        return "greeting"

    # 2. Prediction intent (checked before knowledge since "flagged" overlaps)
    if any(kw in normalized for kw in PREDICTION_KEYWORDS):
        # "why is the mix flagged" is prediction-related if it references results
        print(f"[ROUTER] prediction  <- '{query}'")
        return "prediction"

    # 3. Analytical intent — numerical data queries
    has_number = bool(re.search(r"\b\d+(\.\d+)?\b", normalized))
    has_analytical_kw = any(kw in normalized for kw in ANALYTICAL_KEYWORDS)
    if has_analytical_kw:
        print(f"[ROUTER] analytical  <- '{query}'")
        return "analytical"

    # 4. Knowledge intent — check for construction domain keywords
    is_construction = any(term in normalized for term in CONSTRUCTION_TERMS)
    starts_with_knowledge = any(normalized.startswith(p) for p in KNOWLEDGE_PREFIXES)
    if is_construction or starts_with_knowledge:
        print(f"[ROUTER] knowledge  <- '{query}'")
        return "knowledge"

    # 5. Off-topic
    print(f"[ROUTER] off_topic  <- '{query}'")
    return "off_topic"
