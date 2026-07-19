"""
knowledge_pipeline/llm_engine.py
----------------------------------
Qwen-powered intent router and answer generator.

The Qwen model is the ACTUAL brain of the assistant.
It:
  1. Understands what the user is asking (any phrasing, any format)
  2. Receives retrieved knowledge + live project context as context
  3. Generates a professional, structured engineering answer

No hardcoded regex routing. No pattern matching.
Qwen decides what the user wants and how to answer it.
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

IMPORTANT ENGINEERING DEFINITIONS (always use these, never contradict them):
- HONEYCOMBING: Voids, cavities, or porous pockets in hardened concrete caused by insufficient compaction/vibration, leaving coarse aggregate without mortar between particles. It is NOT cracking. Causes: poor vibration, stiff mix, formwork gaps, reinforcement congestion.
- PLASTIC SHRINKAGE CRACKING: Surface cracks that form in fresh concrete before it hardens, caused by rapid evaporation exceeding bleeding rate. Triggered by high temperature, low humidity, or high wind.
- WATER-CEMENT RATIO (W/C): Ratio of mass of water to mass of cement. Lower W/C = higher strength and lower permeability. IS 456:2000 Table 5 sets maximum W/C per exposure condition.
- CURING: Process of maintaining moisture and temperature after placing to enable cement hydration. IS 456:2000 Section 13.5 requires minimum 14 days for concrete with mineral admixtures.
- IS 456:2000: Indian Standard for Plain and Reinforced Concrete, the primary Indian concrete design code.
- REBOUND HAMMER (IS 13311 Part 2): NDT tool that estimates surface hardness/strength by measuring rebound of a spring-driven hammer.
- UPV (IS 13311 Part 1): Ultrasonic Pulse Velocity test measuring pulse travel time through concrete to assess homogeneity and estimate strength.
"""


@st.cache_resource(show_spinner="Loading AI model...")
def _load_qwen():
    """Load Qwen 2.5-0.5B-Instruct once and cache it for the session."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        return None, str(e)


def _build_context(act_vals: dict, chunks: list[dict], prediction_history: list) -> str:
    """
    Build the context string that Qwen uses to answer.
    Includes: live project parameters + relevant retrieved knowledge.
    """
    lines = []

    # --- Live project data ---
    lines.append("=== ACTIVE PROJECT DATA ===")
    lines.append(f"Concrete Grade: {act_vals.get('concrete_grade', 'M35')}")
    lines.append(f"W/C Ratio (Design): {act_vals.get('water_cement_ratio_design', 0.40):.2f}")
    lines.append(f"W/C Ratio (Actual Site): {act_vals.get('water_cement_ratio_actual', 0.45):.2f}")
    lines.append(f"Curing Duration: {act_vals.get('actual_curing_duration_days', 8)} days")
    lines.append(f"Placing Temperature: {act_vals.get('pour_temp', 30)}°C")
    lines.append(f"Relative Humidity: {act_vals.get('humidity', 50)}%")
    lines.append(f"Wind Exposure: {act_vals.get('wind_exposure', 'Normal')}")

    # Flag computation
    wc_act = act_vals.get('water_cement_ratio_actual', 0.45)
    wc_des = act_vals.get('water_cement_ratio_design', 0.40)
    curing = act_vals.get('actual_curing_duration_days', 8)
    temp   = act_vals.get('pour_temp', 30)
    grade  = act_vals.get('concrete_grade', 'M35')

    flags = []
    if wc_act > wc_des + 0.01:
        flags.append(f"W/C ratio {wc_act:.2f} exceeds design {wc_des:.2f} by {wc_act-wc_des:.2f}")
    if curing < 14:
        flags.append(f"Curing {curing} days is {14-curing} days below IS 456:2000 minimum of 14 days")
    if temp > 30:
        flags.append(f"Placing temperature {temp}°C exceeds IS 7861 hot weather threshold of 30°C")

    if flags:
        lines.append(f"ACTIVE FLAGS ({len(flags)} issues):")
        for f in flags:
            lines.append(f"  - {f}")
    else:
        lines.append("STATUS: All parameters within acceptable ranges")

    # --- Historical predictions ---
    if prediction_history and len(prediction_history) >= 1:
        lines.append("\n=== PREDICTION HISTORY (last 3) ===")
        for h in prediction_history[-3:]:
            lines.append(
                f"- {h.get('timestamp', 'Previous')}: "
                f"Grade={h.get('grade','?')}, "
                f"W/C={h.get('wc_actual','?')}, "
                f"Crack Probability={h.get('prob','?')}"
            )

    # --- Retrieved knowledge chunks ---
    if chunks:
        lines.append("\n=== RETRIEVED ENGINEERING KNOWLEDGE ===")
        seen = set()
        for chunk in chunks[:4]:
            doc_id = chunk.get("doc_id", "")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            title = chunk.get("title", "Document")
            text  = chunk.get("text", "")[:500].strip()
            lines.append(f"[Source: {title}]\n{text}")

    return "\n".join(lines)


def qwen_answer(
    query: str,
    act_vals: dict,
    chunks: list[dict],
    prediction_history: list | None = None
) -> str:
    """
    Main entry point. Uses Qwen to understand the query and generate an answer.
    Falls back to a minimal rule-based response if Qwen fails to load.
    """
    if prediction_history is None:
        prediction_history = []

    tokenizer, model = _load_qwen()

    # Qwen failed to load — use simple fallback
    if tokenizer is None:
        error_msg = model  # contains error string when loading failed
        return (
            f"The AI model could not be loaded ({error_msg}). "
            f"Please check your internet connection and try again."
        )

    context = _build_context(act_vals, chunks, prediction_history)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nUser Question: {query}"}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
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

        # Strip the prompt tokens, keep only new generated tokens
        new_ids = [
            out[len(inp):]
            for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

        if not response:
            return "I could not generate a response. Please rephrase your question."

        return response

    except Exception as e:
        return f"An error occurred while generating a response: {str(e)}"
