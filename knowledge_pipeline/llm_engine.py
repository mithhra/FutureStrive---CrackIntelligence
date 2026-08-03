"""
knowledge_pipeline/llm_engine.py
----------------------------------
Unified Qwen-powered answer generator for the Construction Intelligence Platform.
Handles both Crack Intelligence and Defect Volume Intelligence modules.

Pipeline: Reason -> Retrieve -> Answer
  1. Intent is classified by intent_router.py before this is called
  2. Context is built from the ACTIVE module's live data + FAISS chunks
  3. Qwen generates a structured, grounded response

Reactive extension (image analysis):
  qwen_crack_image_report() — takes structured detector output from crack_detector.py
  and generates a full IS-code-referenced engineering report.
"""

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """You are a senior construction engineer and AI assistant embedded in a Construction Intelligence Platform.

Your role:
- Understand exactly what the user is asking, regardless of how they phrase it
- If the user says "hi", "hello", or small talk, respond with a warm brief greeting and list what you can help with
- If the question is completely unrelated to construction (e.g. sports, cooking, general knowledge), politely say you are specialised in construction engineering only
- Answer using ONLY the project data and retrieved knowledge provided in the context below
- Never invent facts. If the context does not contain the answer, say so clearly
- Always answer the user's actual question directly in the first sentence
- EVERY answer must begin with a short executive summary
- Organize responses into clearly separated sections using markdown headers
- Use professional engineering tone, bullet points for lists, bold key values
- Keep responses focused (150-350 words max)

IMPORTANT ENGINEERING DEFINITIONS — CRACK INTELLIGENCE:
- HONEYCOMBING: Voids, cavities, or porous pockets in hardened concrete caused by insufficient compaction/vibration. It is NOT cracking. Causes: poor vibration, stiff mix, formwork gaps, reinforcement congestion.
- PLASTIC SHRINKAGE CRACKING: Surface cracks that form in fresh concrete before it hardens, caused by rapid evaporation exceeding bleeding rate. Triggered by high temperature, low humidity, or high wind.
- WATER-CEMENT RATIO (W/C): Ratio of mass of water to mass of cement. Lower W/C = higher strength and lower permeability. IS 456:2000 Table 5 sets maximum W/C per exposure condition.
- CURING: Maintaining moisture and temperature after placing to enable cement hydration. IS 456:2000 Section 13.5 requires minimum 14 days for concrete with mineral admixtures.
- IS 456:2000: Indian Standard for Plain and Reinforced Concrete — the primary Indian concrete design code.
- REBOUND HAMMER (IS 13311 Part 2): NDT tool estimating surface hardness/strength by measuring rebound of a spring-driven hammer.
- UPV (IS 13311 Part 1): Ultrasonic Pulse Velocity test measuring pulse travel time through concrete to assess homogeneity and strength.

IMPORTANT ENGINEERING DEFINITIONS — DEFECT VOLUME INTELLIGENCE:
- QC HOLD POINT: A mandatory inspection checkpoint in the construction sequence that must be cleared before work proceeds to the next activity. Low compliance = higher defect risk.
- SPI (Schedule Performance Index): Ratio of earned value to planned value. SPI < 1 = behind schedule. Rushed work under schedule pressure directly increases defect rates.
- SUBCONTRACTOR CLASS: A/B/C grading based on past performance. Class C contractors historically produce 2-4x more defects per floor than Class A.
- DEFECT RATE PER FLOOR: The historical average number of defects raised per floor on a project. A rising trend across floors signals systemic process or material failure.
- THIRD-PARTY INSPECTION: Independent QA audit by a client-appointed inspector. Projects with third-party involvement typically have 30-50% lower defect escape rates.
- SKILL RATIO: Fraction of the workforce classified as skilled (vs unskilled/semi-skilled). Lower ratio directly correlates with higher incidence of process-related defects.
- DEFECT SEVERITY GRADES: Minor (cosmetic, no structural impact) / Moderate (functional impact, rework required) / Major (structural or regulatory concern) / Critical (safety risk, stop-work required).
"""


@st.cache_resource(show_spinner="Loading AI model...")
def _load_qwen():
    """Load Qwen 2.5-0.5B-Instruct once and cache it for the session."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        return None, str(e)


def _build_crack_context(act_vals: dict, chunks: list, prediction_history: list) -> str:
    """Build context block for Crack Intelligence queries."""
    lines = ["=== ACTIVE CRACK INTELLIGENCE DATA ==="]
    lines.append(f"Concrete Grade         : {act_vals.get('concrete_grade', 'M35')}")
    lines.append(f"W/C Ratio (Design)     : {act_vals.get('water_cement_ratio_design', 0.40):.2f}")
    lines.append(f"W/C Ratio (Actual)     : {act_vals.get('water_cement_ratio_actual', 0.45):.2f}")
    lines.append(f"Curing Duration        : {act_vals.get('actual_curing_duration_days', 8)} days")
    lines.append(f"Placing Temperature    : {act_vals.get('pour_temp', 30)}C")
    lines.append(f"Relative Humidity      : {act_vals.get('humidity', 50)}%")
    lines.append(f"Wind Exposure          : {act_vals.get('wind_exposure', 'Normal')}")

    wc_act = act_vals.get('water_cement_ratio_actual', 0.45)
    wc_des = act_vals.get('water_cement_ratio_design', 0.40)
    curing = act_vals.get('actual_curing_duration_days', 8)
    temp   = act_vals.get('pour_temp', 30)
    grade  = act_vals.get('concrete_grade', 'M35')
    max_wc = {"M25": 0.55, "M30": 0.50, "M35": 0.45, "M40": 0.40}.get(grade, 0.45)

    flags = []
    if wc_act > wc_des + 0.01:
        flags.append(f"W/C ratio {wc_act:.2f} exceeds design {wc_des:.2f} by {wc_act-wc_des:.2f} (IS 456 max for {grade}: {max_wc:.2f})")
    if curing < 14:
        flags.append(f"Curing {curing} days is {14-curing} days below IS 456:2000 Section 13.5 minimum (14 days)")
    if temp > 30:
        flags.append(f"Placing temperature {temp}C exceeds IS 7861 hot weather threshold of 30C")

    if flags:
        lines.append(f"ACTIVE FLAGS ({len(flags)} issues):")
        for f in flags:
            lines.append(f"  - {f}")
    else:
        lines.append("STATUS: All crack parameters within acceptable ranges")

    if prediction_history:
        crack_hist = [h for h in prediction_history if h.get("module") == "crack"][-3:]
        if crack_hist:
            lines.append("\n=== RECENT CRACK PREDICTIONS ===")
            for h in crack_hist:
                lines.append(f"- {h.get('timestamp','?')}: Grade={h.get('grade','?')}, "
                             f"W/C={h.get('wc_actual','?')}, Probability={h.get('prob','?'):.0%}, "
                             f"Severity={h.get('severity','?')}, Type={h.get('type','?')}")

    return "\n".join(lines)


def _build_defect_context(act_vals: dict, chunks: list, prediction_history: list) -> str:
    """Build context block for Defect Volume Intelligence queries."""
    lines = ["=== ACTIVE DEFECT VOLUME DATA ==="]
    lines.append(f"Subcontractor Class    : {act_vals.get('subcontractor_class', 'Class B')}")
    lines.append(f"Past Defect Rate/Floor : {act_vals.get('past_defect_rate_per_floor', 3.0):.1f}")
    lines.append(f"Skill Ratio            : {act_vals.get('skill_ratio', 0.70):.0%}")
    lines.append(f"QC Hold Compliance     : {act_vals.get('qc_hold_point_compliance_pct', 0.80):.0%}")
    lines.append(f"SPI                    : {act_vals.get('spi', 1.00):.2f}")
    lines.append(f"Non-Productive Days    : {act_vals.get('non_productive_days', 0)}")
    lines.append(f"Approved Supplier      : {'Yes' if act_vals.get('approved_supplier', 1) else 'No'}")
    lines.append(f"Test Certificate       : {act_vals.get('test_certificate_status', 'Pass')}")
    lines.append(f"Construction Stage     : {act_vals.get('construction_stage', 'Superstructure')}")
    lines.append(f"Third-Party Inspection : {'Yes' if act_vals.get('third_party_inspection', 0) else 'No'}")

    pred = act_vals.get("last_defect_prediction", {})
    if pred:
        lines.append("\n=== LAST DEFECT PREDICTION RESULT ===")
        lines.append(f"Predicted Defect Count : {pred.get('count', '?')}")
        lines.append(f"Dominant Defect Type   : {pred.get('type', '?')}")
        lines.append(f"Severity Grade         : {pred.get('severity', '?')}")
        lines.append(f"Root Cause             : {pred.get('root_cause', '?')}")

    return "\n".join(lines)


def _build_context(act_vals: dict, chunks: list, prediction_history: list,
                   active_module: str = "crack") -> str:
    """Build the full context string passed to Qwen, based on active module."""
    if active_module == "defect":
        ctx = _build_defect_context(act_vals, chunks, prediction_history)
    else:
        ctx = _build_crack_context(act_vals, chunks, prediction_history)

    # Append retrieved knowledge chunks (shared for both modules)
    if chunks:
        ctx += "\n\n=== RETRIEVED ENGINEERING KNOWLEDGE ==="
        seen = set()
        for chunk in chunks[:4]:
            doc_id = chunk.get("doc_id", "")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            title = chunk.get("title", "Document")
            text  = chunk.get("text", "")[:500].strip()
            ctx += f"\n[Source: {title}]\n{text}"

    return ctx


def qwen_answer(
    query: str,
    act_vals: dict,
    chunks: list,
    prediction_history: list | None = None,
    active_module: str = "crack"
) -> str:
    """
    Main entry point. Uses Qwen to understand the query and generate an answer.
    active_module: "crack" or "defect" — controls which context block is injected.
    """
    if prediction_history is None:
        prediction_history = []

    tokenizer, model = _load_qwen()

    if tokenizer is None:
        error_msg = model
        return (f"The AI model could not be loaded ({error_msg}). "
                f"Please check your internet connection and try again.")

    context = _build_context(act_vals, chunks, prediction_history, active_module)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nUser Question: {query}"}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=400,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        new_ids = [
            out[len(inp):]
            for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

        return response if response else "I could not generate a response. Please rephrase your question."

    except Exception as e:
        return f"An error occurred while generating a response: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# REACTIVE IMAGE ANALYSIS  — new addition, no changes to existing functions above
# ═══════════════════════════════════════════════════════════════════════════════

# IS 456:2000 max permissible crack widths by crack type
_IS_WIDTH_LIMITS = {
    "Hairline"   : 0.10,
    "Shrinkage"  : 0.20,
    "Structural" : 0.20,
    "Settlement" : 0.30,
}

# Remediation actions by crack type — IS code referenced
_REMEDIATION = {
    "Hairline": [
        "Apply polymer-modified cement slurry or surface sealer (IS 456:2000 clause 13.5)",
        "Monitor with crack width gauge at 7-day intervals",
        "Improve curing on future pours — extend to minimum 14 days (IS 456 Sec 13.5)",
        "Check ambient evaporation rate; install windbreaks if >0.5 kg/m²/hr",
    ],
    "Shrinkage": [
        "Inject cracks with low-viscosity epoxy (IS 13311 compliant)",
        "Control W/C ratio strictly per IS 456:2000 Table 5 limits for the exposure class",
        "Apply IS 12118 curing compound on all exposed surfaces immediately after stripping",
        "Add micro-polypropylene fibres to future mixes to control plastic shrinkage",
    ],
    "Structural": [
        "STOP WORK — do not apply additional loads until structural engineer reviews",
        "Commission IS 516 core extraction for in-situ strength verification",
        "Perform UPV test per IS 13311 Part 1 to assess internal homogeneity",
        "Inject with structural epoxy resin injection system (IS 13311 Part 2)",
        "Review design drawings for reinforcement adequacy — engage structural consultant",
    ],
    "Settlement": [
        "Engage geotechnical engineer immediately — assess foundation/sub-grade conditions",
        "Monitor crack progression with Demec gauge at 48-hour intervals",
        "Do not apply live loads until settlement cause is identified and arrested",
        "Consider ground improvement or underpinning depending on investigation findings",
    ],
}

_URGENCY_LEVELS = {
    "Minor"   : ("LOW",      "Monitor — routine repair at next maintenance cycle"),
    "Moderate": ("MEDIUM",   "Repair within 7 days — assign site engineer to review"),
    "Severe"  : ("HIGH",     "Immediate repair required — restrict access to element"),
    "Critical": ("CRITICAL", "STOP WORK — evacuate area, engage structural engineer now"),
    "None"    : ("NONE",     "No action required"),
}

IMAGE_REPORT_SYSTEM_PROMPT = """You are a senior structural engineer and construction QA specialist.
You are analysing the output of an automated crack detection model applied to a site photograph.

Your job is to:
1. Interpret the detection results in plain engineering language
2. Assess the severity and likely root cause based on crack type
3. Reference applicable Indian Standards (IS 456:2000, IS 13311, IS 516)
4. Provide clear, prioritised remediation steps
5. State the urgency level clearly

Format your response with these sections:
- Executive Summary (2-3 sentences)
- Root Cause Analysis
- Risk to Structure
- Remediation Recommendations (numbered, IS-code referenced)
- Urgency & Next Steps

Be direct. Use engineering terminology. Keep to 250-400 words.
Never say "I cannot determine" — always give your best professional assessment.
"""


def _build_image_report_context(det: dict, chunks: list) -> str:
    """Build the context block passed to Qwen for an image analysis report."""
    crack_type = det.get("crack_type", "Structural")
    severity   = det.get("severity_estimate", "Moderate")
    width_mm   = det.get("estimated_width_mm", 0.0)
    limit_mm   = _IS_WIDTH_LIMITS.get(crack_type, 0.20)
    exceeds    = width_mm > limit_mm

    urgency_code, urgency_desc = _URGENCY_LEVELS.get(severity, ("MEDIUM", "Review required"))
    remed_steps = _REMEDIATION.get(crack_type, _REMEDIATION["Structural"])

    lines = [
        "=== CRACK DETECTION MODEL OUTPUT ===",
        f"Crack Detected       : {'YES' if det.get('crack_detected') else 'NO'}",
        f"Detection Confidence : {det.get('confidence', 0)*100:.1f}%",
        f"Crack Type           : {crack_type}",
        f"Severity Grade       : {severity}",
        f"Area Fraction        : {det.get('area_fraction', 0):.2f}% of image classified as crack",
        f"Number of Instances  : {det.get('num_instances', 0)} separate crack segment(s) detected",
        f"Estimated Width      : {width_mm:.2f} mm",
        f"IS 456 Width Limit   : {limit_mm} mm for {crack_type} cracks",
        f"Limit Exceeded       : {'YES' if exceeds else 'NO'}",
        f"Urgency Level        : {urgency_code} — {urgency_desc}",
        "",
        "=== IS CODE REMEDIATION REFERENCE ===",
    ]
    for i, step in enumerate(remed_steps, 1):
        lines.append(f"  {i}. {step}")

    if chunks:
        lines.append("\n=== RETRIEVED ENGINEERING KNOWLEDGE ===")
        seen = set()
        for chunk in chunks[:3]:
            doc_id = chunk.get("doc_id", "")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            title = chunk.get("title", "Document")
            text  = chunk.get("text", "")[:400].strip()
            lines.append(f"[Source: {title}]\n{text}")

    return "\n".join(lines)


def qwen_crack_image_report(
    det: dict,
    user_query: str = "Analyse this crack image and provide an engineering report.",
    chunks: list = None,
) -> str:
    """
    Generate a structured engineering report from crack detection model output.

    det     : dict returned by crack_detector.run_crack_detection()
    query   : optional user message that accompanied the image upload
    chunks  : optional FAISS retrieved knowledge chunks

    Returns a markdown-formatted engineering report string.
    """
    if chunks is None:
        chunks = []

    # Handle no-crack case with a concise message
    if not det.get("crack_detected"):
        return (
            "## No Crack Detected\n\n"
            f"**Detection confidence: {det.get('confidence', 0)*100:.1f}%**\n\n"
            "The crack detection model did not identify any crack signatures in the "
            "uploaded image. The surface appears intact based on the analysis.\n\n"
            "**Recommended actions:**\n"
            "- If you suspect a crack that was not detected, try uploading a higher "
            "resolution image with direct, even lighting\n"
            "- Ensure the photo is taken perpendicular to the surface with no glare\n"
            "- You may also conduct a manual visual survey per IS 13311 guidelines\n"
        )

    tokenizer, model = _load_qwen()

    if tokenizer is None:
        # LLM not available — generate a rule-based fallback report
        return _fallback_image_report(det)

    context = _build_image_report_context(det, chunks)

    messages = [
        {"role": "system", "content": IMAGE_REPORT_SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Context (crack detection output):\n{context}\n\n"
            f"User note: {user_query}\n\n"
            "Generate a professional engineering report based on the above detection results."
        )}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=450,
                temperature=0.25,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_ids = [
            out[len(inp):]
            for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

        if response:
            # Prepend a structured header the LLM won't always generate itself
            header = _build_report_header(det)
            return header + "\n\n" + response
        return _fallback_image_report(det)

    except Exception as e:
        return _fallback_image_report(det) + f"\n\n> *(LLM error: {str(e)[:80]})*"


def _build_report_header(det: dict) -> str:
    """Build the metrics summary block shown above the LLM narrative."""
    sev  = det.get("severity_estimate", "Moderate")
    crack_type = det.get("crack_type", "Unknown")
    conf       = det.get("confidence", 0) * 100
    area       = det.get("area_fraction", 0)
    instances  = det.get("num_instances", 0)
    width_mm   = det.get("estimated_width_mm", 0.0)
    limit_mm   = _IS_WIDTH_LIMITS.get(crack_type, 0.20)
    mode       = det.get("model_mode", "stub")
    mode_badge = "**DEMO MODE**" if "stub" in mode else "**Live Model**"

    return (
        f"## Reactive Crack Analysis Report ({mode_badge})\n\n"
        f"| Feature | Value |\n"
        f"|---|---|\n"
        f"| Crack Detected | YES |\n"
        f"| Detection Confidence | **{conf:.1f}%** |\n"
        f"| Crack Type | **{crack_type}** |\n"
        f"| Severity | **{sev}** |\n"
        f"| Area Fraction | {area:.2f}% of image |\n"
        f"| No. of Instances | {instances} |\n"
        f"| Est. Crack Width | {width_mm:.2f} mm |\n"
        f"| IS 456 Width Limit | {limit_mm} mm for {crack_type} |\n"
        f"| Limit Exceeded | {'**YES**' if width_mm > limit_mm else 'No'} |\n"
    )


def _fallback_image_report(det: dict) -> str:
    """
    Rule-based engineering report when Qwen is unavailable.
    Generates a complete, IS-code-referenced report from the detection dict alone.
    """
    crack_type = det.get("crack_type", "Structural")
    severity   = det.get("severity_estimate", "Moderate")
    width_mm   = det.get("estimated_width_mm", 0.0)
    limit_mm   = _IS_WIDTH_LIMITS.get(crack_type, 0.20)
    exceeds    = width_mm > limit_mm
    urgency_code, urgency_desc = _URGENCY_LEVELS.get(severity, ("MEDIUM", "Review required"))
    remed_steps = _REMEDIATION.get(crack_type, _REMEDIATION["Structural"])

    root_causes = {
        "Hairline"  : "Usually caused by thermal expansion/contraction, early drying shrinkage, or minor formwork movement. Typically non-structural.",
        "Shrinkage" : "Caused by rapid moisture loss from fresh concrete surface exceeding the bleeding rate. Linked to high W/C ratio, hot weather, low humidity, or high wind exposure (IS 7861).",
        "Structural": "Indicates inadequate flexural reinforcement, overloading beyond design limits, foundation settlement, or premature formwork stripping. Requires structural assessment.",
        "Settlement": "Sub-grade or foundation movement. Can indicate consolidation settlement, bearing capacity failure, or differential settlement between elements.",
    }

    risk_statements = {
        "Minor"   : "Low risk to structural integrity. No immediate action required but monitor to ensure crack width does not propagate.",
        "Moderate": "Moderate risk. The crack may allow water ingress leading to reinforcement corrosion over time if not sealed. Repair within 7 days.",
        "Severe"  : "High risk. Crack width exceeds IS 456:2000 permissible limits. Restrict access to element and initiate repairs immediately.",
        "Critical": "Critical risk. Structure may be compromised. Do not apply any load. Evacuate and engage a structural engineer immediately.",
    }

    header = _build_report_header(det)

    body = (
        f"### Executive Summary\n"
        f"The detection model identified a **{crack_type}** crack of **{severity}** severity "
        f"with {det.get('confidence', 0)*100:.1f}% confidence. "
        f"Estimated width is **{width_mm:.2f} mm** against an IS 456:2000 permissible limit of {limit_mm} mm. "
        f"{'The limit is **exceeded** — immediate attention required.' if exceeds else 'Width is within permissible limits.'}\n\n"

        f"### Root Cause Analysis\n"
        f"{root_causes.get(crack_type, 'Root cause requires site investigation.')}\n\n"

        f"### Risk to Structure\n"
        f"{risk_statements.get(severity, 'Risk assessment requires structural engineer review.')}\n\n"

        f"### Remediation Recommendations\n"
        + "\n".join(f"{i}. {step}" for i, step in enumerate(remed_steps, 1)) +

        f"\n\n### Urgency & Next Steps\n"
        f"**Urgency Level: {urgency_code}**  —  {urgency_desc}\n\n"
        f"- Document crack with scale reference and date stamp\n"
        f"- Log in the site defect register with geo-tag and photo\n"
        f"- Re-inspect at 7-day intervals to track propagation rate\n"
    )

    return header + "\n" + body
