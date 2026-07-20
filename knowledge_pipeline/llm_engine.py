"""
knowledge_pipeline/llm_engine.py
----------------------------------
Unified Qwen-powered answer generator for the Construction Intelligence Platform.
Handles both Crack Intelligence and Defect Volume Intelligence modules.

Pipeline: Reason -> Retrieve -> Answer
  1. Intent is classified by intent_router.py before this is called
  2. Context is built from the ACTIVE module's live data + FAISS chunks
  3. Qwen generates a structured, grounded response
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
