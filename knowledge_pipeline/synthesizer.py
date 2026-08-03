"""
knowledge_pipeline/synthesizer.py
-----------------------------------
Reason → Retrieve → Answer (RAR) Pipeline
==========================================

The assistant NEVER leads with retrieved documents.
It reasons about the user's intent, answers from project context first,
then uses retrieved chunks only as supporting evidence.

Pipeline:
  Step 1 — understand_intent()      : What is the user actually asking?
  Step 2 — inspect_project_context(): What project data answers this?
  Step 3 — generate_direct_answer() : Answer the question from context
  Step 4 — select_supporting_docs() : Pick only docs that back the answer
  Step 5 — compose_response()       : Assemble final structured response
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INTENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "definition":    [r"\bwhat is\b", r"\bdefine\b", r"\bexplain what\b", r"\bmean(ing)?\b", r"\bwhat (are|does)\b"],
    "root_cause":    [r"\bwhy (is|are|was|were|did|does)\b", r"\bwhat caused?\b", r"\breason for\b", r"\bflagged\b", r"\broot cause\b"],
    "recommendation":[r"\bhow (to|can|do|should)\b", r"\bwhat should\b", r"\brecommend\b", r"\bprevent\b", r"\bfix\b", r"\bimprove\b", r"\bwhat (steps|action)\b"],
    "prediction":    [r"\bpredict\b", r"\bforecast\b", r"\bwill (it|this|the)\b", r"\bprobability\b", r"\brisk\b"],
    "comparison":    [r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bdifference between\b", r"\bwhich is better\b"],
    "simulation":    [r"\bsimulat\b", r"\bwhat (happens?|if)\b", r"\bif (i|we|curing|w/c)\b", r"\bincreas\b.{1,20}days?\b"],
    "report":        [r"\bgenerate (a )?report\b", r"\bsummariz\b", r"\binspection report\b"],
    "explanation":   [r"\bexplain\b", r"\bhow does\b", r"\bwhy does\b", r"\bdescribe\b"],
    "requirement":   [r"\brequire(ment|d)?\b", r"\bspec(ification)?\b", r"\bstandard\b", r"\bcode\b", r"\bis 456\b", r"\bcpwd\b"],
}

DOMAIN_PATTERNS = {
    "crack":         [r"\bcrack\b", r"\bcracki?ng\b", r"\bshrinkage\b", r"\bplastic crack\b"],
    "curing":        [r"\bcuring\b", r"\bcure\b", r"\bhydration\b", r"\bcuring days?\b", r"\bcuring compound\b"],
    "water_cement":  [r"\bwater.?cement\b", r"\bw/?c\b", r"\bmix water\b", r"\bw/c ratio\b"],
    "honeycombing":  [r"\bhoneycomb\b", r"\bvoid(s)?\b", r"\bcompaction\b", r"\bvibration\b", r"\bporous\b"],
    "temperature":   [r"\btemperature\b", r"\bhot weather\b", r"\bplacing temp\b", r"\bdegrees?\b"],
    "concrete_grade":[r"\bgrade\b", r"\bm\d{2}\b", r"\bcompressive strength\b", r"\bfck\b"],
    "safety":        [r"\bsafety\b", r"\bppe\b", r"\bhazard\b", r"\bformwork\b", r"\bfall\b"],
    "inspection":    [r"\bndt\b", r"\brebound hammer\b", r"\bupv\b", r"\bultrasonic\b", r"\bcover meter\b"],
    "cost":          [r"\bcost\b", r"\bbudget\b", r"\boverrun\b", r"\bboq\b"],
    "delay":         [r"\bdelay\b", r"\bschedule\b", r"\bcritical path\b"],
    "material":      [r"\bcemen(t|titious)\b", r"\baggregate\b", r"\badmixture\b", r"\bsteel\b", r"\brebar\b"],
    "standards":     [r"\bis 456\b", r"\bcpwd\b", r"\bnptel\b", r"\bcode\b", r"\bclause\b"],
}


def understand_intent(query: str) -> dict:
    """
    Classify user intent and domain before any retrieval.
    Returns {intent, domains, is_project_specific, question_word}
    """
    q = query.lower().strip()

    # Detect question word
    qword = "general"
    for word in ["why", "what", "how", "compare", "predict", "simulate", "explain",
                 "generate", "list", "show", "define", "describe"]:
        if q.startswith(word) or f" {word} " in q:
            qword = word
            break

    # Classify intent
    intent = "general"
    for label, patterns in INTENT_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            intent = label
            break

    # Classify domains (can be multi-label)
    domains = []
    for domain, patterns in DOMAIN_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            domains.append(domain)

    # Check if the question is project-specific
    project_keywords = ["current", "my", "this", "project", "mix", "flagged",
                        "our", "active", "the pour", "today", "predicted"]
    is_project_specific = any(kw in q for kw in project_keywords)

    return {
        "intent": intent,
        "domains": domains if domains else ["general"],
        "question_word": qword,
        "is_project_specific": is_project_specific,
        "raw_query": query,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — INSPECT PROJECT CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

# Engineering thresholds (IS 456:2000, IS 7861, ACI 305)
THRESHOLDS = {
    "max_wc_M20": 0.55, "max_wc_M25": 0.50, "max_wc_M30": 0.45,
    "max_wc_M35": 0.45, "max_wc_M40": 0.40,
    "min_curing_opc": 7, "min_curing_ppc": 10, "min_curing_admixture": 14,
    "max_placing_temp": 38, "hot_weather_threshold": 30,
    "critical_evap_rate": 1.0,  # kg/m2/hr
    "low_humidity_threshold": 50,
}


def inspect_project_context(act_vals: dict) -> dict:
    """
    Analyse the active project parameters and flag any deviations from standards.
    Returns a structured dict of findings used to answer BEFORE retrieval.
    """
    grade   = act_vals.get("concrete_grade", "M35")
    wc_act  = act_vals.get("water_cement_ratio_actual", 0.45)
    wc_des  = act_vals.get("water_cement_ratio_design", 0.40)
    curing  = act_vals.get("actual_curing_duration_days", 8)
    temp    = act_vals.get("pour_temp", 30)
    hum     = act_vals.get("humidity", 50)
    wind    = act_vals.get("wind_exposure", "Normal")

    # Get max allowable W/C for grade
    max_wc_key = f"max_wc_{grade}"
    max_wc = THRESHOLDS.get(max_wc_key, 0.45)

    flags = []
    wc_deviation = wc_act - wc_des
    wc_vs_code   = wc_act - max_wc

    if wc_deviation > 0.02:
        flags.append({
            "parameter": "W/C Ratio",
            "value": f"{wc_act:.2f}",
            "threshold": f"{wc_des:.2f} (design) / {max_wc:.2f} (IS 456 max for {grade})",
            "deviation": f"+{wc_deviation:.2f} above design",
            "severity": "HIGH" if wc_deviation > 0.05 else "MEDIUM",
            "reason": (
                f"Site W/C ratio {wc_act:.2f} exceeds design target {wc_des:.2f} by {wc_deviation:.2f}. "
                f"Each 0.05 increase in W/C ratio reduces 28-day strength by approximately 5–7 MPa "
                f"and increases permeability, raising crack and durability risk."
            )
        })

    if wc_vs_code > 0:
        flags.append({
            "parameter": "IS 456 W/C Compliance",
            "value": f"{wc_act:.2f}",
            "threshold": f"{max_wc:.2f} (IS 456:2000 Table 5 for {grade})",
            "deviation": f"+{wc_vs_code:.2f} above code limit",
            "severity": "HIGH",
            "reason": (
                f"W/C ratio {wc_act:.2f} exceeds the IS 456:2000 maximum of {max_wc:.2f} "
                f"for {grade} concrete. This is a code non-compliance issue."
            )
        })

    min_curing = THRESHOLDS["min_curing_admixture"]
    if curing < min_curing:
        deficit = min_curing - curing
        flags.append({
            "parameter": "Curing Duration",
            "value": f"{curing} days",
            "threshold": f"{min_curing} days (IS 456:2000 Section 13.5)",
            "deviation": f"{deficit} days short",
            "severity": "HIGH" if curing < 7 else "MEDIUM",
            "reason": (
                f"Curing of {curing} days is {deficit} days below the IS 456:2000 minimum of "
                f"{min_curing} days for {grade} concrete with admixtures. "
                f"Inadequate curing reduces strength by 20–40% and significantly increases "
                f"drying shrinkage crack probability."
            )
        })

    if temp > THRESHOLDS["hot_weather_threshold"]:
        flags.append({
            "parameter": "Placing Temperature",
            "value": f"{temp}°C",
            "threshold": f"{THRESHOLDS['hot_weather_threshold']}°C (IS 7861 hot weather threshold)",
            "deviation": f"+{temp - THRESHOLDS['hot_weather_threshold']}°C above threshold",
            "severity": "HIGH" if temp > 35 else "MEDIUM",
            "reason": (
                f"Placing temperature of {temp}°C exceeds the IS 7861 hot weather threshold of 30°C. "
                f"Elevated temperature accelerates cement hydration, reduces workability retention, "
                f"and dramatically increases plastic shrinkage cracking risk — especially combined "
                f"with {hum}% relative humidity."
            )
        })

    if hum < THRESHOLDS["low_humidity_threshold"] or wind == "Exposed":
        # Estimate evaporation rate (simplified ACI 305 formula)
        evap_estimate = max(0, (temp - 18) * 0.05 - hum * 0.008 + (0.3 if wind == "Exposed" else 0.1))
        if evap_estimate > THRESHOLDS["critical_evap_rate"] or hum < 40:
            flags.append({
                "parameter": "Plastic Shrinkage Risk",
                "value": f"Humidity {hum}%, Wind: {wind}",
                "threshold": "Evaporation rate must be < 1.0 kg/m²/hr",
                "deviation": "High evaporation risk conditions",
                "severity": "HIGH",
                "reason": (
                    f"The combination of {hum}% relative humidity and {wind.lower()} wind exposure "
                    f"creates conditions likely exceeding the 1.0 kg/m²/hr evaporation threshold "
                    f"at which plastic shrinkage cracks become almost inevitable. "
                    f"Fog misting and windbreaks are critical."
                )
            })

    return {
        "grade": grade,
        "wc_actual": wc_act,
        "wc_design": wc_des,
        "wc_max_code": max_wc,
        "curing_days": curing,
        "placing_temp": temp,
        "humidity": hum,
        "wind": wind,
        "flags": flags,
        "has_flags": len(flags) > 0,
        "flag_count": len(flags),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — GENERATE DIRECT ANSWER (from reasoning, before retrieval)
# ─────────────────────────────────────────────────────────────────────────────

def generate_direct_answer(intent_info: dict, ctx: dict) -> str:
    """
    Answer the user's actual question FIRST, using project context and
    engineering knowledge. Documents are not consulted at this step.
    """
    intent  = intent_info["intent"]
    domains = intent_info["domains"]
    qword   = intent_info["question_word"]
    query   = intent_info["raw_query"].lower()
    flags   = ctx["flags"]

    grade  = ctx["grade"]
    wc_act = ctx["wc_actual"]
    wc_des = ctx["wc_design"]
    curing = ctx["curing_days"]
    temp   = ctx["placing_temp"]
    hum    = ctx["humidity"]
    wind   = ctx["wind"]

    # ── Root Cause / Why flagged ───────────────────────────────────────────
    if intent == "root_cause" or "flagged" in query or "why" in query:
        if not ctx["has_flags"]:
            return (
                f"The current {grade} mix is within acceptable parameters for all monitored indicators. "
                f"W/C ratio ({wc_act:.2f}) is at design target, curing ({curing} days) meets IS 456:2000 "
                f"minimums, and placing conditions are within safe operating ranges."
            )
        reasons = []
        for f in flags:
            reasons.append(f"**{f['parameter']}** ({f['value']}) — {f['reason']}")
        flag_summary = "\n\n".join(reasons)
        return (
            f"The {grade} mix is flagged due to **{ctx['flag_count']} parameter deviation(s)** "
            f"from engineering thresholds:\n\n{flag_summary}"
        )

    # ── Curing question ────────────────────────────────────────────────────
    if "curing" in domains:
        if intent == "simulation" or "increase" in query or "what happens" in query:
            cur_new = curing + 2
            improvement = min((14 - curing) * 5, 25) if curing < 14 else 0
            return (
                f"Increasing curing from **{curing} days** to **{cur_new} days** on your {grade} project "
                f"would reduce drying shrinkage crack risk by an estimated {improvement}%. "
                f"At {cur_new} days, the concrete would still be {max(0, 14 - cur_new)} days short of the "
                f"IS 456:2000 minimum of 14 days for {grade} mixes. The primary benefit is improved "
                f"surface hardness and reduced permeability — both of which directly reduce long-term "
                f"chloride ingress and corrosion risk."
            )
        if intent in ("requirement", "explanation") or "is 456" in query or "spec" in query:
            return (
                f"Under IS 456:2000 Section 13.5, the minimum curing duration for {grade} concrete is:\n\n"
                f"- **OPC only:** 7 days\n"
                f"- **PPC or blended cement:** 10 days\n"
                f"- **With mineral admixtures (fly ash, GGBS):** 14 days\n"
                f"- **Hot weather (>30°C) or aggressive exposure:** 14 days minimum\n\n"
                f"Your project currently shows **{curing} days** of curing. "
                f"{'This meets the 7-day OPC minimum but not the 14-day admixture requirement.' if 7 <= curing < 14 else 'This is below even the 7-day minimum and requires immediate extension.'}"
            )
        if intent == "recommendation" or "how" in query:
            return (
                f"To meet IS 456:2000 curing requirements for your {grade} mix, "
                f"extend curing from the current **{curing} days** by **{max(0, 14 - curing)} more days** "
                f"to reach the 14-day minimum. Apply a curing compound (IS 12118 compliant) within "
                f"20–30 minutes of finishing. At a placing temperature of {temp}°C and humidity of {hum}%, "
                f"wet burlap or hessian covering is also recommended to reduce surface evaporation."
            )

    # ── Water-cement ratio ─────────────────────────────────────────────────
    if "water_cement" in domains:
        if intent == "root_cause" or "why" in query:
            dev = wc_act - wc_des
            strength_loss = dev / 0.05 * 6  # ~6 MPa per 0.05 W/C increase
            return (
                f"The site W/C ratio of **{wc_act:.2f}** exceeds the design target of **{wc_des:.2f}** "
                f"by **{dev:.2f}** — an exceedance of {dev/wc_des*100:.0f}%. "
                f"This is most commonly caused by unauthorized water additions at the batching plant "
                f"or on site, or by wet aggregates whose moisture content was not accounted for in "
                f"the mix water calculation. The practical impact is an estimated reduction of "
                f"**{strength_loss:.0f} MPa** in 28-day compressive strength and significantly "
                f"increased permeability — raising both crack risk and long-term durability concerns."
            )
        if intent == "recommendation":
            return (
                f"To bring the W/C ratio from **{wc_act:.2f}** back to the design target of **{wc_des:.2f}**:\n\n"
                f"1. **Stop all on-site water additions** immediately — this is the single most common cause\n"
                f"2. **Test aggregate moisture** (IS 2386 Part 3) before every batch — wet fine aggregate "
                f"contains 3–7% free water that counts toward the mix W/C\n"
                f"3. **Recalculate batch water** using current aggregate moisture values in the IS 10262 mix design\n"
                f"4. **Increase site supervision** at delivery point — reject any truck where concrete "
                f"appears excessively fluid or slump exceeds target by more than 25mm"
            )
        return (
            f"The water-cement ratio is the ratio of the mass of water to the mass of cement in the mix. "
            f"It is the primary determinant of concrete strength and durability. "
            f"Your {grade} mix has a design W/C of **{wc_des:.2f}** with a site-measured actual of **{wc_act:.2f}**. "
            f"IS 456:2000 Table 5 specifies a maximum W/C of **{ctx['wc_max_code']:.2f}** for {grade} concrete. "
            f"{'Your current ratio is within code limits.' if wc_act <= ctx['wc_max_code'] else 'Your current ratio exceeds the code maximum — this is a non-compliance issue.'}"
        )

    # ── Honeycombing ───────────────────────────────────────────────────────
    if "honeycombing" in domains:
        if intent == "definition" or "what is" in query:
            return (
                "Honeycombing refers to voids, cavities, or rough porous pockets in hardened concrete "
                "caused by the absence of mortar between coarse aggregate particles. It results in "
                "reduced structural strength, increased permeability, and exposed reinforcement — "
                "all of which accelerate corrosion and structural deterioration."
            )
        if intent == "root_cause" or "cause" in query or "why" in query:
            return (
                "Honeycombing is caused by one or more of the following:\n\n"
                "1. **Inadequate vibration** — vibrator not inserted at regular 500mm centres, "
                "or withdrawn too quickly (>5 cm/sec). Most common cause on site.\n"
                "2. **Mix too stiff** — W/C ratio too low without plasticizer; concrete cannot flow around bars\n"
                "3. **Formwork gaps** — mortar leaks through joints, leaving aggregate-only zones\n"
                "4. **Reinforcement congestion** — bar spacing less than 3× the maximum aggregate size\n"
                "5. **Excessive free-fall height** — concrete dropped more than 1,500mm causes segregation"
            )
        if intent == "recommendation" or "fix" in query or "repair" in query:
            return (
                "Honeycombing repair depends on severity:\n\n"
                "**Minor** (depth < 25mm, area < 0.1m²):\n"
                "- Chip to solid substrate → apply SBR bonding agent → fill with 1:2 cement:sand mortar → cure 7 days\n\n"
                "**Moderate** (25–75mm deep):\n"
                "- Form sides with temporary shuttering → fill with non-shrink cementitious grout\n\n"
                "**Severe** (>75mm deep or reinforcement exposed):\n"
                "- Structural engineer assessment required → epoxy injection (IS 13938) → "
                "full core replacement with high-strength micro-concrete"
            )

    # ── Temperature ────────────────────────────────────────────────────────
    if "temperature" in domains:
        return (
            f"Your concrete was placed at **{temp}°C**. "
            f"{'This exceeds the IS 7861 hot weather threshold of 30°C. ' if temp > 30 else 'This is within the safe placing temperature range. '}"
            f"At {temp}°C with {hum}% relative humidity, the rate of moisture evaporation from the "
            f"fresh concrete surface is {'elevated — increasing plastic shrinkage crack risk significantly. ' if temp > 30 else 'manageable with standard curing precautions. '}"
            f"IS 7861 requires precooling of mix water and aggregates when placing temperature exceeds 35°C, "
            f"and prohibits placing above 38°C without special measures."
        )

    # ── Safety ────────────────────────────────────────────────────────────
    if "safety" in domains:
        return (
            "Concrete construction safety requirements under NBC Part 7 and IS 3696 include:\n\n"
            "- **Formwork:** Must be designed by a structural engineer for 24–25 kN/m³ concrete weight "
            "plus 2.4 kN/m² live load\n"
            "- **PPE:** Chemical-resistant gloves mandatory (concrete pH 12–13 causes delayed burns), "
            "steel-capped boots, hard hat at all times\n"
            "- **Edge protection:** Guardrail + mid-rail + kickboard for all working platforms above 1.8m\n"
            "- **Pump lines:** Never disconnect under pressure — depressurize completely first\n"
            "- **Hot weather:** Provide cool drinking water, schedule heavy work before 11am"
        )

    # ── Standards / IS 456 ─────────────────────────────────────────────────
    if "standards" in domains or intent == "requirement":
        return (
            f"IS 456:2000 specifies the following for {grade} concrete:\n\n"
            f"| Requirement | Value |\n|---|---|\n"
            f"| Maximum W/C ratio | {ctx['wc_max_code']:.2f} |\n"
            f"| Minimum curing duration | 14 days (with admixtures) |\n"
            f"| Minimum cover (Severe exposure) | 45mm |\n"
            f"| Minimum cement content | 320 kg/m³ (Severe) |\n"
            f"| Characteristic strength | {grade[1:]} MPa at 28 days |\n\n"
            f"Your project: W/C actual **{wc_act:.2f}** "
            f"({'within limit' if wc_act <= ctx['wc_max_code'] else 'EXCEEDS limit'}), "
            f"curing **{curing} days** ({'adequate' if curing >= 14 else 'insufficient'})."
        )

    # -- General / fallback
    _fc = ctx['flag_count']
    _hf = ctx['has_flags']
    flag_msg = f'there are **{_fc} active parameter flags** that require attention.' if _hf else 'all parameters are within acceptable ranges.'
    return (
        f'Based on your active {grade} project context '
        f'W/C actual {wc_act:.2f} (design: {wc_des:.2f}), '
        f'curing {curing} days, placing temperature {temp}C, humidity {hum}% -- '
        f'{flag_msg}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SELECT SUPPORTING DOCUMENTS (after answer is written)
# ─────────────────────────────────────────────────────────────────────────────

def select_supporting_docs(chunks: list[dict], intent_info: dict, ctx: dict) -> list[dict]:
    """
    From retrieved chunks, pick only those that SUPPORT the answer already given.
    Discard chunks that are off-topic or would add noise.
    """
    if not chunks:
        return []

    domains   = set(intent_info["domains"])
    intent    = intent_info["intent"]
    query_lc  = intent_info["raw_query"].lower()

    # Map intent/domain to which chunk domains are actually relevant
    RELEVANT_DOMAIN_MAP = {
        "curing":        {"concrete", "cracks", "standards"},
        "water_cement":  {"concrete", "cracks", "quality"},
        "honeycombing":  {"honeycombing", "quality", "inspection"},
        "temperature":   {"concrete", "cracks", "quality"},
        "safety":        {"safety"},
        "crack":         {"cracks", "concrete", "standards"},
        "inspection":    {"inspection", "quality"},
        "cost":          {"cost"},
        "delay":         {"delays"},
        "material":      {"material_management"},
        "standards":     {"standards", "concrete"},
        "general":       set(),  # accept all
    }

    relevant_domains = set()
    for d in domains:
        relevant_domains |= RELEVANT_DOMAIN_MAP.get(d, set())
    if not relevant_domains:
        relevant_domains = None  # accept all

    selected = []
    seen_docs = set()

    for chunk in chunks:
        chunk_domain = chunk.get("domain", "")
        doc_id       = chunk.get("doc_id", "")

        # Skip already-seen documents (take only one chunk per doc)
        if doc_id in seen_docs:
            continue

        # Filter by domain relevance
        if relevant_domains and chunk_domain not in relevant_domains:
            continue

        # Quick relevance check: chunk text must contain at least one query keyword
        chunk_text_lc = chunk.get("text", "").lower()
        query_words = set(w for w in re.findall(r'\b\w{4,}\b', query_lc))
        if query_words and not any(w in chunk_text_lc for w in list(query_words)[:8]):
            continue

        selected.append(chunk)
        seen_docs.add(doc_id)

        if len(selected) >= 3:  # maximum 3 supporting docs
            break

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — COMPOSE FINAL RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

_STANDARDS_RE = re.compile(
    r"\b(IS\s*\d+[\w.:]*|NBC\s*\d+|ACI\s*\d+[\w.]*|CPWD[\w.\s]*|"
    r"OSHA[\w.\s]*|FHWA[\w.-]*|ASTM\s*[A-Z]\d+|IS\s*456|IS\s*10262|"
    r"IS\s*13311|IS\s*516|IS\s*4082|IS\s*7861|IS\s*12118)\b",
    re.IGNORECASE
)

_NOISE_RE = re.compile(r"\s{2,}|_{3,}|\[\d+\]")

STANDARD_DESCRIPTIONS = {
    "IS 456":   "IS 456:2000 — Plain and Reinforced Concrete Code of Practice (Bureau of Indian Standards)",
    "IS 7861":  "IS 7861 — Hot and Cold Weather Concreting (BIS)",
    "IS 10262": "IS 10262:2019 — Concrete Mix Design Guidelines (BIS)",
    "IS 13311": "IS 13311 — Non-Destructive Testing of Concrete: UPV (Part 1) and Rebound Hammer (Part 2)",
    "IS 516":   "IS 516:2018 — Method of Tests for Strength of Concrete",
    "IS 4082":  "IS 4082 — Recommendations for Stacking and Storage of Materials",
    "IS 12118": "IS 12118 — Specification for Liquid Membrane-Forming Curing Compounds",
    "IS 1786":  "IS 1786:2008 — High Strength Deformed Steel Bars for Concrete Reinforcement",
    "IS 3696":  "IS 3696 — Safety Code for Scaffolding",
    "CPWD":     "CPWD Specifications 2019 — Central Public Works Department Technical Specifications",
    "NBC":      "National Building Code of India 2016 — Part 7: Construction Management and Safety",
    "NPTEL":    "NPTEL Course Materials — IIT Madras / IIT Kharagpur Concrete Technology & Maintenance",
}


def _extract_supporting_evidence(docs: list[dict]) -> str:
    """Extract 1–2 key sentences per document that support the already-given answer."""
    if not docs:
        return ""

    lines = []
    for doc in docs:
        text  = doc.get("text", "")
        title = doc.get("title", "")
        org   = doc.get("source_org", "")
        year  = doc.get("pub_year", "")
        domain= doc.get("domain", "")

        # Clean noise
        text = _NOISE_RE.sub(" ", text).strip()

        # Extract 1–2 meaningful sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = []
        for sent in sentences:
            sent = sent.strip()
            if 40 < len(sent) < 350 and not sent.isupper():
                kept.append(sent)
            if len(kept) == 2:
                break

        if kept:
            evidence = " ".join(kept)
            # Cap at 300 chars to avoid walls of text
            if len(evidence) > 300:
                evidence = evidence[:300].rsplit(" ", 1)[0] + "."
            lines.append(f"According to **{title}** ({org}, {year}): _{evidence}_")

    return "\n\n".join(lines)


def _generate_recommendations(intent_info: dict, ctx: dict) -> str:
    """Generate actionable bullet-point recommendations."""
    domains = intent_info["domains"]
    flags   = ctx["flags"]
    grade   = ctx["grade"]
    wc_act  = ctx["wc_actual"]
    wc_des  = ctx["wc_design"]
    curing  = ctx["curing_days"]
    temp    = ctx["placing_temp"]
    hum     = ctx["humidity"]
    wind    = ctx["wind"]

    bullets = []

    # Flag-driven recommendations
    for flag in flags:
        if flag["parameter"] == "W/C Ratio":
            bullets.append(
                f"**Reduce site W/C to {wc_des:.2f}** — eliminate on-site water additions; "
                f"test aggregate moisture (IS 2386 Part 3) before every batch"
            )
        elif flag["parameter"] == "Curing Duration":
            deficit = max(0, 14 - curing)
            bullets.append(
                f"**Extend curing by {deficit} days** to reach 14-day IS 456 minimum — "
                f"apply IS 12118-compliant curing compound within 20–30 min of finishing"
            )
        elif flag["parameter"] == "Placing Temperature":
            bullets.append(
                f"**Hot weather precautions:** Pre-cool mix water with ice; schedule pours before 7am or after 5pm; "
                f"use a retarder for pours where transit time exceeds 45 minutes (IS 7861)"
            )
        elif flag["parameter"] == "Plastic Shrinkage Risk":
            bullets.append(
                f"**Install windbreaks** on all exposed faces; apply fog mist to surface before finishing; "
                f"deploy curing compound immediately after floating (within 20 minutes)"
            )

    # Domain-specific additions
    if "honeycombing" in domains:
        bullets += [
            "**Check vibrator insertion spacing** — maximum 500mm centres; minimum 5 seconds per insertion point",
            "**Inspect formwork joints** before each pour for gaps that allow mortar loss",
        ]
    if "inspection" in domains:
        bullets += [
            f"**Conduct rebound hammer survey** (IS 13311 Part 2) across the poured element at 300mm grid",
            "**Perform UPV testing** (IS 13311 Part 1) — velocities below 3,500 m/s indicate quality concerns",
        ]
    if "safety" in domains:
        bullets += [
            "**Verify formwork propping** meets design loads before any pour",
            "**Mandatory PPE:** chemical-resistant gloves, steel-capped boots, hard hat for all personnel",
        ]

    if not bullets:
        bullets = [
            f"Maintain W/C at design target of {wc_des:.2f} — reject any delivery with slump more than 25mm above target",
            f"Continue monitoring curing daily and extend to 14 days minimum",
            f"Document all cube test results and cross-reference against IS 456:2000 acceptance criteria",
        ]

    return "\n".join(f"- {b}" for b in bullets[:6])


def _build_references(docs: list[dict]) -> str:
    """Build a clean reference list from unique source documents."""
    seen = set()
    refs = []
    for doc in docs:
        key = doc.get("doc_id", "")
        if key not in seen:
            seen.add(key)
            title = doc.get("title", "Unknown")
            org   = doc.get("source_org", "")
            year  = doc.get("pub_year", "")
            refs.append(f"- {title}" + (f" — {org}, {year}" if org else ""))
    return "\n".join(refs) if refs else "- IS 456:2000 — Bureau of Indian Standards"


def _suggested_actions(intent_info: dict, ctx: dict) -> str:
    """Smart follow-up suggestions based on what was asked and what flags exist."""
    actions = set()
    intent  = intent_info["intent"]
    domains = intent_info["domains"]
    flags   = ctx["flags"]

    if ctx["has_flags"]:
        actions.add("Run Crack Risk Prediction with updated parameters")
        actions.add("View SHAP Analysis to rank contributing factors")
    if "curing" in domains or any(f["parameter"] == "Curing Duration" for f in flags):
        actions.add("Simulate extended curing duration impact")
    if "water_cement" in domains or any(f["parameter"] == "W/C Ratio" for f in flags):
        actions.add("Recalculate IS 10262 Mix Design")
    if "honeycombing" in domains:
        actions.add("Generate Honeycombing Inspection Report")
    if "inspection" in domains:
        actions.add("Plan NDT Survey (Rebound Hammer + UPV)")
    if "crack" in domains:
        actions.add("Run Root Cause Analysis")
        actions.add("Run Crack Risk Prediction")
    if intent == "root_cause":
        actions.add("View SHAP Analysis")
    if intent == "recommendation":
        actions.add("Compare with Similar Historical Projects")

    # Always add a universal option
    actions.add("Run Crack Risk Prediction")

    return "\n".join(f"- {a}" for a in sorted(actions)[:5])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compose_response(
    query: str,
    chunks: list[dict],
    act_vals: dict,
    prediction_history: list | None = None
) -> str:
    """
    Reason → Retrieve → Answer pipeline.

    1. Understand intent (no retrieval yet)
    2. Inspect project context (no retrieval yet)
    3. Generate direct answer from reasoning
    4. Select only relevant supporting docs
    5. Compose structured final response
    """
    if prediction_history is None:
        prediction_history = []

    # Step 1 — Understand intent
    intent_info = understand_intent(query)

    # Step 2 — Inspect project context
    ctx = inspect_project_context(act_vals)

    # Step 3 — Direct answer first (before any document text)
    direct_answer = generate_direct_answer(intent_info, ctx)

    # Step 4 — Select only relevant supporting docs
    supporting_docs = select_supporting_docs(chunks, intent_info, ctx)
    evidence_text   = _extract_supporting_evidence(supporting_docs)

    # Step 5 — Compose final response in the required order
    sections = []

    # ── 1. Direct Answer ──────────────────────────────────────────────────
    sections.append("## Direct Answer\n")
    sections.append(direct_answer)

    # ── 2. Why / Engineering Reasoning ───────────────────────────────────
    if ctx["has_flags"] and intent_info["intent"] not in ("definition",):
        sections.append("\n\n---\n\n## Engineering Analysis\n")
        rows = [f"| {f['parameter']} | {f['value']} | {f['threshold']} | {f['severity']} |"
                for f in ctx["flags"]]
        table = (
            "| Parameter | Current Value | Threshold | Severity |\n"
            "|---|---|---|---|\n"
        ) + "\n".join(rows)
        sections.append(table)

    # ── 3. Supporting Evidence ─────────────────────────────────────────────
    if evidence_text:
        sections.append("\n\n---\n\n## Supporting Evidence\n")
        sections.append(evidence_text)

    # ── 4. Historical Context ─────────────────────────────────────────────
    if prediction_history and len(prediction_history) >= 2:
        wc_act = ctx["wc_actual"]
        similar = [h for h in prediction_history if abs(h.get("wc_actual", 0) - wc_act) <= 0.05]
        if similar:
            avg_prob = sum(h.get("prob", 0) for h in similar) / len(similar)
            sections.append("\n\n---\n\n## Historical Context\n")
            sections.append(
                f"In **{len(similar)} previous prediction(s)** with a similar W/C ratio (~{wc_act:.2f}), "
                f"the average crack probability was **{avg_prob*100:.0f}%**. "
                f"_These are based on prior organizational predictions, not the current analysis._"
            )

    # ── 5. Recommendations ────────────────────────────────────────────────
    sections.append("\n\n---\n\n## Recommendations\n")
    sections.append(_generate_recommendations(intent_info, ctx))

    # ── 6. References ─────────────────────────────────────────────────────
    if supporting_docs:
        sections.append("\n\n---\n\n## References\n")
        sections.append(_build_references(supporting_docs))

    # ── 7. Suggested Next Actions ─────────────────────────────────────────
    sections.append("\n\n---\n\n## Suggested Next Actions\n")
    sections.append(_suggested_actions(intent_info, ctx))

    return "".join(sections)
