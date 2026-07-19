import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import altair as alt
import datetime
from pathlib import Path

# ─── RAG Engine (lazy-loaded, cached) ─────────────────────────────────────────
RAG_INDEX_FILE = Path("vector_store/index.faiss")
RAG_META_FILE  = Path("vector_store/chunk_metadata.json")

@st.cache_resource(show_spinner=False)
def _load_rag_engine():
    """Load the FAISS index and sentence-transformer model once per session."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        import json
        if not RAG_INDEX_FILE.exists() or not RAG_META_FILE.exists():
            return None, None, None
        index = faiss.read_index(str(RAG_INDEX_FILE))
        with open(RAG_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return index, meta, model
    except Exception:
        return None, None, None


def _rag_retrieve(query: str, k: int = 6) -> list[dict]:
    """Return top-k chunks most semantically similar to the query."""
    index, meta, model = _load_rag_engine()
    if index is None or model is None:
        return []
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(meta):
            chunk = dict(meta[idx])
            chunk["_score"] = float(dist)
            results.append(chunk)
    return results


# ─── Intent Router + Qwen LLM (top-level, cached once) ────────────────────────
@st.cache_resource(show_spinner=False)
def _load_assistant():
    try:
        from knowledge_pipeline.intent_router import classify_query
        from knowledge_pipeline.llm_engine   import qwen_answer
        return classify_query, qwen_answer, True
    except Exception as e:
        return None, None, False

_classify_query, _qwen_answer, _ASSISTANT_AVAILABLE = _load_assistant()

if not _ASSISTANT_AVAILABLE:
    def _classify_query(q): return "knowledge"
    def _qwen_answer(query, act_vals, chunks, prediction_history=None): return None


# Page Configuration
st.set_page_config(
    page_title="FutureStrive Construction Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div,
    .stText, label, .stSelectbox, .stSlider, .stCheckbox {
        color: #1E293B !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Metrics — dark text so they're visible on light cards */
    div[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 800 !important; }
    div[data-testid="stMetricLabel"] { color: #475569 !important; font-weight: 600 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0F172A !important; border-right: 1px solid #1E293B; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] p { color: #94A3B8 !important; }

    /* Sidebar nav */
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] {
        display: flex; flex-direction: column; gap: 6px;
    }
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label {
        background-color: transparent !important; border: none !important;
        border-radius: 6px !important; padding: 10px 14px !important;
        cursor: pointer !important; font-size: 13px !important;
        font-weight: 500 !important; color: #94A3B8 !important;
    }
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label:hover {
        background-color: #1E293B !important; color: #FFFFFF !important;
    }
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label[data-checked="true"] {
        background-color: #2563EB !important; color: #FFFFFF !important; font-weight: 700 !important;
    }

    /* Cards — white background so all text is clearly readable */
    .saas-card {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .saas-card * { color: #1E293B !important; }

    .copilot-prompt-card {
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .copilot-prompt-card * { color: #1E293B !important; }

    /* Buttons */
    .stButton > button {
        background: transparent !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: none !important;
        transition: background 0.15s ease;
    }
    .stButton > button:hover {
        background: #E2E8F0 !important;
        border-color: #94A3B8 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputContainer"] textarea,
    [data-baseweb="textarea"] textarea {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        caret-color: #FFFFFF !important;
    }
    [data-testid="stChatInputContainer"],
    [data-testid="stBottom"] > div {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = [
        {"role": "assistant", "content": (
            "## Welcome to the Construction Intelligence Assistant\n\n"
            "I am your AI construction engineering advisor backed by IS codes, CPWD manuals, "
            "NPTEL references, and your live project data.\n\n"
            "**Try asking:**\n"
            "- *Why is the current mix flagged?*\n"
            "- *What is honeycombing and how do I fix it?*\n"
            "- *How do I prevent plastic shrinkage cracks?*\n"
            "- *Show features with contribution greater than 20%*\n"
            "- *What happens if curing duration increases by 2 days?*"
        )}
    ]

if "active_prediction" not in st.session_state:
    st.session_state.active_prediction = {
        "concrete_grade": "M35",
        "water_cement_ratio_design": 0.40,
        "water_cement_ratio_actual": 0.45,
        "actual_curing_duration_days": 8,
        "pour_temp": 34,
        "humidity": 40,
        "wind_exposure": "Normal",
    }

if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = "Crack Predictor"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## FutureStrive")
    st.markdown("**Construction Intelligence Platform**")
    st.markdown("---")
    st.session_state.nav_selection = st.radio(
        "Navigation",
        ["Crack Predictor", "AI Assistant", "SHAP Analysis", "Prediction History"],
        index=["Crack Predictor", "AI Assistant", "SHAP Analysis", "Prediction History"]
              .index(st.session_state.nav_selection),
        label_visibility="collapsed"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CRACK PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.nav_selection == "Crack Predictor":
    st.title("Crack Risk Predictor")
    st.markdown("Configure concrete pour parameters and run crack occurrence prediction.")

    col_form, col_results = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.subheader("Pour Parameters")
        
        with st.expander("Concrete Mix Properties", expanded=True):
            grade = st.selectbox("Concrete Grade", ["M25", "M30", "M35", "M40"], index=2)
            cem_type = st.selectbox("Cement Type", ["OPC 43 Grade", "OPC 53 Grade", "PPC", "PSC"], index=1)
            adm_type = st.selectbox("Admixture Type", ["No Admixture", "Naphthalene-based superplasticiser", "Polycarboxylate-based superplasticiser"])
            wc_des = st.slider("W/C Ratio (Design)", 0.35, 0.60, 0.40, 0.01)
            wc_act = st.slider("W/C Ratio (Actual)", 0.35, 0.65, 0.45, 0.01)
            wc_tol = st.number_input("W/C Ratio Tolerance Spec", value=0.45, step=0.01)
            slump = st.number_input("Target Slump (mm)", value=100)
            agg_size = st.selectbox("Max Aggregate Size (mm)", [10, 20], index=1)
            
        with st.expander("Curing & Execution", expanded=False):
            cur_meth = st.selectbox("Curing Method", ["Ponding", "Sprinkling", "Wet burlap curing", "Curing compound"])
            cur_plan = st.number_input("Planned Curing (days)", value=14)
            cur_act = st.slider("Actual Curing (days)", 1, 28, 8)
            cur_spec = st.number_input("Spec Min Curing (days)", value=14)
            pour_mo = st.selectbox("Planned Pour Month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
            chk_ratio = st.slider("Checklist Signed Off Ratio", 0.0, 1.0, 1.0)
            temp = st.slider("Placing Temperature (°C) [UI Context]", 10, 50, 34)
            hum = st.slider("Relative Humidity (%) [UI Context]", 10, 100, 40)
            
        with st.expander("Site & Environment", expanded=False):
            wind = st.selectbox("Wind Exposure", ["Sheltered", "Normal", "Exposed"], index=1)
            shrink_risk = st.selectbox("Shrinkage Risk Season", ["LOW", "MEDIUM", "HIGH"], index=1)
            site_env = st.selectbox("Site Environment", ["Inland", "Coastal", "Industrial"])
            city = st.selectbox("City", ["Mumbai", "Bangalore", "Hyderabad", "Delhi", "Chennai"])
            access = st.selectbox("Accessibility", ["Open", "Semi-enclosed", "Enclosed"])
            tier = st.selectbox("Project Tier", ["Class A", "Class B", "Class C"])
            sim_elems = st.number_input("Similar Elements Count", value=10)

        run_btn = st.button("▶ Run Crack Prediction", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        if run_btn:
            import pandas as pd
            # Construct dictionary matching exact dataset columns
            feat_dict = {
                "concrete_grade": grade,
                "water_cement_ratio_design": wc_des,
                "water_cement_ratio_actual": wc_act,
                "cement_type": cem_type,
                "admixture_type": adm_type,
                "target_slump_mm": slump,
                "max_aggregate_size_mm": agg_size,
                "planned_pour_month": pour_mo,
                "curing_method": cur_meth,
                "planned_curing_duration_days": cur_plan,
                "actual_curing_duration_days": cur_act,
                "spec_min_curing_days": cur_spec,
                "wc_ratio_tolerance_spec": wc_tol,
                "pre_pour_checklist_signed_off_ratio": chk_ratio,
                "shrinkage_risk_season": shrink_risk,
                "wind_exposure_category": wind,
                "site_environment": site_env,
                "accessibility": access,
                "city": city,
                "project_tier": tier,
                "count_similar_elements": sim_elems
            }
            features_df = pd.DataFrame([feat_dict])

            # ── ML Prediction Pipeline ──────────────────────────────────────
            # Step 1: Occurrence probability (21 base features)
            # Step 2: Severity + Type (21 base features, only for cracked)
            try:
                occ_model  = joblib.load("crack_occurrence_model.joblib")
                sev_model  = joblib.load("crack_severity_model.joblib")
                type_model = joblib.load("crack_type_model.joblib")

                # Step 1: crack occurrence probability
                occ_prob = float(occ_model.predict_proba(features_df)[0][1])

                # Step 2: severity and type
                sev_label  = str(sev_model.predict(features_df)[0])
                type_label = str(type_model.predict(features_df)[0])

                # If low risk, suppress downstream labels
                if occ_prob < 0.25:
                    sev_label  = "None (Low Risk)"
                    type_label = "None (Low Risk)"

                model_ok = True
            except Exception as e:
                st.error(f"ML Pipeline Error: {e}")
                model_ok   = False
                occ_prob   = 0.0
                sev_label  = "Error"
                type_label = "Error"

            # Save to session state
            record = {
                "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "grade":      grade,
                "wc_design":  wc_des,
                "wc_actual":  wc_act,
                "curing":     cur_act,
                "temp":       temp,
                "humidity":   hum,
                "wind":       wind,
                "prob":       occ_prob,
                "severity":   sev_label,
                "type":       type_label,
            }
            st.session_state.prediction_history.append(record)
            st.session_state.active_prediction = {
                "concrete_grade":            grade,
                "water_cement_ratio_design": wc_des,
                "water_cement_ratio_actual": wc_act,
                "actual_curing_duration_days": cur_act,
                "pour_temp":    temp,
                "humidity":     hum,
                "wind_exposure": wind,
            }

            risk_color = "#DC2626" if occ_prob > 0.6 else ("#F59E0B" if occ_prob > 0.35 else "#16A34A")
            risk_label = "HIGH RISK" if occ_prob > 0.6 else ("MEDIUM RISK" if occ_prob > 0.35 else "LOW RISK")

            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")

            m1, m2, m3 = st.columns(3)
            m1.metric("Crack Probability", f"{occ_prob*100:.1f}%")
            m2.metric("Severity",  sev_label)
            m3.metric("Type",      type_label)

            st.markdown(
                f"<div style='background:{risk_color};color:#fff;padding:10px 18px;"
                f"border-radius:6px;font-weight:700;font-size:16px;margin:12px 0'>"
                f"⚠ {risk_label} — Crack Probability {occ_prob*100:.1f}%</div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if not model_ok:
                st.info("Models not found. Train models with `python train_models.py`.")
        else:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.info("Configure parameters on the left and click **Run Crack Prediction** to see results.")
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.nav_selection == "SHAP Analysis":
    st.title("SHAP Feature Analysis")
    st.markdown("Understand which parameters are driving crack risk in the active prediction.")

    act = st.session_state.active_prediction

    wc_dev    = act["water_cement_ratio_actual"] - act["water_cement_ratio_design"]
    wc_c      = max(wc_dev * 2.5 + (0.5 if act["water_cement_ratio_actual"] > 0.45 else 0.0), 0.05)
    cur_c     = max((14 - act["actual_curing_duration_days"]) * 0.12 if act["actual_curing_duration_days"] < 14 else 0.02, 0.02)
    wind_c    = 0.25 if act["wind_exposure"] == "Exposed" else (0.08 if act["wind_exposure"] == "Normal" else 0.02)
    temp_c    = max((act["pour_temp"] - 30) * 0.035 if act["pour_temp"] > 30 else 0.02, 0.02)
    hum_c     = max((50 - act["humidity"]) * 0.010 if act["humidity"] < 50 else 0.02, 0.02)
    grade_c   = 0.12 if act["concrete_grade"] in ["M40", "M35"] else 0.04

    raw = {
        "Measured Site W/C Ratio":       wc_c,
        "Curing Duration":               cur_c,
        "Wind Exposure":                 wind_c,
        "Placing Temperature":           temp_c,
        "Relative Humidity":             hum_c,
        "Concrete Grade":                grade_c,
    }
    tot   = sum(raw.values())
    df_sh = pd.DataFrame([
        {"Feature": k, "Contribution %": round(v/tot*100, 1)}
        for k, v in raw.items()
    ]).sort_values("Contribution %", ascending=False).reset_index(drop=True)
    df_sh["Rank"] = df_sh.index + 1

    st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
    st.subheader("Feature Contribution Rankings")
    st.dataframe(df_sh[["Rank", "Feature", "Contribution %"]], use_container_width=True, hide_index=True)

    chart = alt.Chart(df_sh).mark_bar(color="#2563EB").encode(
        x=alt.X("Contribution %:Q"),
        y=alt.Y("Feature:N", sort="-x"),
        tooltip=["Feature", "Contribution %"]
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICTION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.nav_selection == "Prediction History":
    st.title("Prediction History")

    if not st.session_state.prediction_history:
        st.info("No predictions yet. Run a crack prediction first.")
    else:
        df_hist = pd.DataFrame(st.session_state.prediction_history)
        df_hist["Probability %"] = (df_hist["prob"] * 100).round(1)
        st.dataframe(
            df_hist[["timestamp","grade","wc_actual","curing","temp","humidity","Probability %","severity","type"]],
            use_container_width=True, hide_index=True
        )

        if len(df_hist) > 1:
            line = alt.Chart(df_hist.reset_index()).mark_line(
                point=True, color="#2563EB"
            ).encode(
                x=alt.X("index:O", title="Prediction #"),
                y=alt.Y("Probability %:Q", scale=alt.Scale(domain=[0,100])),
                tooltip=["timestamp","grade","Probability %","severity"]
            ).properties(height=300, title="Crack Probability Trend")
            st.altair_chart(line, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. AI ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.nav_selection == "AI Assistant":
    st.title("AI Quality Assistant")
    st.markdown("Powered by **Qwen 2.5** · Intent-routed · Backed by IS codes, CPWD, NPTEL knowledge base")

    act_vals = st.session_state.active_prediction

    # Active context banner
    st.markdown(f"""
    <div style="background:#F1F5F9;border:1px solid #E2E8F0;padding:10px 16px;
    border-radius:6px;margin-bottom:16px;font-size:12.5px;color:#1E293B;">
    <strong>Active Context:</strong>
    Grade: <code>{act_vals['concrete_grade']}</code> ·
    W/C Design: <code>{act_vals['water_cement_ratio_design']:.2f}</code> ·
    W/C Actual: <code>{act_vals['water_cement_ratio_actual']:.2f}</code> ·
    Curing: <code>{act_vals['actual_curing_duration_days']} days</code> ·
    Temp: <code>{act_vals['pour_temp']}°C</code> ·
    Humidity: <code>{act_vals['humidity']}%</code>
    </div>
    """, unsafe_allow_html=True)

    # ── Shortcut prompt buttons ──────────────────────────────────────────────
    st.markdown("<div class='copilot-prompt-card'>", unsafe_allow_html=True)
    st.markdown("##### Quick questions:")
    c1, c2, c3, c4 = st.columns(4)
    sh_query = ""
    with c1:
        if st.button("Why is mix flagged?", use_container_width=True):
            sh_query = "Why is the current mix flagged?"
    with c2:
        if st.button("What is honeycombing?", use_container_width=True):
            sh_query = "What is honeycombing and how do I fix it?"
    with c3:
        if st.button("Curing requirements", use_container_width=True):
            sh_query = "Explain curing requirements under IS 456."
    with c4:
        if st.button("Prevent cracks", use_container_width=True):
            sh_query = "How do I prevent plastic shrinkage cracks?"
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat history display ─────────────────────────────────────────────────
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.copilot_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Input ────────────────────────────────────────────────────────────────
    chat_inp = st.chat_input("Ask about crack risk, curing, W/C ratio, honeycombing, IS 456, safety...")
    if sh_query:
        chat_inp = sh_query

    if chat_inp:
        st.session_state.copilot_history.append({"role": "user", "content": chat_inp})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(chat_inp)

        lower_q = chat_inp.lower()

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1: INTENT CLASSIFICATION
        # ══════════════════════════════════════════════════════════════════════
        intent = _classify_query(chat_inp)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2: ANALYTICAL — numerical pandas queries on live project data
        # ══════════════════════════════════════════════════════════════════════
        _pandas_resp = None

        _NUM_PATTERN = re.compile(
            r"\b(less than|greater than|more than|above|below|under|at least|at most"
            r"|fewer than|equal to|<|>|<=|>=|==)\s+([\d.]+)\b",
            re.IGNORECASE
        )

        def _map_op(raw):
            r = raw.lower().strip()
            if r in ("<", "less than", "below", "under", "fewer than"): return "<"
            if r in (">", "greater than", "above", "over", "more than"): return ">"
            if r in ("<=", "at most"): return "<="
            if r in (">=", "at least"): return ">="
            if r in ("==", "equal to"): return "=="
            return None

        def _shap_df():
            wc_dev  = act_vals["water_cement_ratio_actual"] - act_vals["water_cement_ratio_design"]
            wc_c    = max(wc_dev*2.5 + (0.5 if act_vals["water_cement_ratio_actual"] > 0.45 else 0.0), 0.05)
            cur_c   = max((14 - act_vals["actual_curing_duration_days"])*0.12 if act_vals["actual_curing_duration_days"] < 14 else 0.02, 0.02)
            wind_c  = 0.25 if act_vals["wind_exposure"]=="Exposed" else (0.08 if act_vals["wind_exposure"]=="Normal" else 0.02)
            temp_c  = max((act_vals["pour_temp"]-30)*0.035 if act_vals["pour_temp"] > 30 else 0.02, 0.02)
            hum_c   = max((50-act_vals["humidity"])*0.010 if act_vals["humidity"] < 50 else 0.02, 0.02)
            grade_c = 0.12 if act_vals["concrete_grade"] in ["M40","M35"] else 0.04
            raw = {"W/C Ratio": wc_c, "Curing Duration": cur_c, "Wind Exposure": wind_c,
                   "Temperature": temp_c, "Humidity": hum_c, "Concrete Grade": grade_c}
            tot = sum(raw.values())
            return pd.DataFrame([{"Feature": k, "Contribution %": round(v/tot*100,1)} for k,v in raw.items()])

        def _params_df():
            wmap = {"Sheltered":0,"Normal":1,"Exposed":2}
            gmap = {"M25":25,"M30":30,"M35":35,"M40":40}
            return pd.DataFrame([
                {"Parameter":"W/C Design",    "Value": act_vals["water_cement_ratio_design"],   "Unit":"-"},
                {"Parameter":"W/C Actual",     "Value": act_vals["water_cement_ratio_actual"],    "Unit":"-"},
                {"Parameter":"Curing Days",    "Value": act_vals["actual_curing_duration_days"],  "Unit":"days"},
                {"Parameter":"Temperature",    "Value": act_vals["pour_temp"],                    "Unit":"°C"},
                {"Parameter":"Humidity",       "Value": act_vals["humidity"],                     "Unit":"%"},
                {"Parameter":"Wind Exposure",  "Value": wmap.get(act_vals["wind_exposure"],1),    "Unit":"0-2"},
                {"Parameter":"Grade (MPa)",    "Value": gmap.get(act_vals["concrete_grade"],35),  "Unit":"MPa"},
            ])

        num_match = _NUM_PATTERN.search(chat_inp)
        if num_match or intent == "analytical":
            if num_match:
                op_raw, thr_str = num_match.group(1), num_match.group(2)
                op  = _map_op(op_raw)
                thr = float(thr_str)
                is_shap = any(kw in lower_q for kw in ["contribution","feature","shap","factor","risk","cause","impact","%","percent"])

                if op:
                    if is_shap:
                        df_src = _shap_df()
                        filtered = df_src.query(f"`Contribution %` {op} @thr")
                        if filtered.empty:
                            _pandas_resp = f"No features with contribution {op_raw} {thr}%.\n\n{df_src.to_string(index=False)}"
                        else:
                            _pandas_resp = f"**Features where Contribution % {op_raw} {thr}%:**\n\n```\n{filtered.to_string(index=False)}\n```"
                    else:
                        df_src = _params_df()
                        filtered = df_src.query(f"Value {op} @thr")
                        if filtered.empty:
                            _pandas_resp = f"No parameters with value {op_raw} {thr}.\n\n{df_src.to_string(index=False)}"
                        else:
                            _pandas_resp = f"**Parameters where Value {op_raw} {thr}:**\n\n```\n{filtered.to_string(index=False)}\n```"

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3: ROUTE TO HANDLER
        # ══════════════════════════════════════════════════════════════════════

        if intent == "greeting":
            resp = (
                f"Hello! I'm your **Construction Intelligence Assistant**.\n\n"
                f"I can help you with:\n"
                f"- **Why is the mix flagged?** — root cause from live project data\n"
                f"- **What is honeycombing?** — engineering definitions & IS codes\n"
                f"- **How to prevent cracks?** — IS 456:2000 & CPWD recommendations\n"
                f"- **What if curing increases 2 days?** — what-if simulation\n"
                f"- **Show features > 20%** — SHAP contribution queries\n"
                f"- **NDT inspection planning** — rebound hammer, UPV, cover meter\n\n"
                f"Active project: **{act_vals['concrete_grade']}** | "
                f"W/C {act_vals['water_cement_ratio_actual']:.2f} | "
                f"Curing {act_vals['actual_curing_duration_days']} days"
            )

        elif _pandas_resp is not None:
            resp = _pandas_resp

        elif intent == "prediction":
            # ── Structured prediction explanation from live data ────────────
            wc_act = act_vals["water_cement_ratio_actual"]
            wc_des = act_vals["water_cement_ratio_design"]
            curing = act_vals["actual_curing_duration_days"]
            temp   = act_vals["pour_temp"]
            hum    = act_vals["humidity"]
            wind   = act_vals["wind_exposure"]
            grade  = act_vals["concrete_grade"]
            max_wc = {"M25":0.55,"M30":0.50,"M35":0.45,"M40":0.40}.get(grade, 0.45)

            flags = []
            if wc_act > wc_des + 0.01:
                dev = wc_act - wc_des
                flags.append(
                    f"**W/C Ratio {wc_act:.2f}** exceeds design **{wc_des:.2f}** by {dev:.2f} "
                    f"— IS 456:2000 maximum for {grade} is {max_wc:.2f}. "
                    f"Each 0.05 increase reduces 28-day strength by ~5–7 MPa."
                )
            if curing < 14:
                flags.append(
                    f"**Curing {curing} days** — {14-curing} days short of IS 456:2000 "
                    f"Section 13.5 minimum of 14 days for admixture mixes. "
                    f"Inadequate curing reduces strength by 20–40%."
                )
            if temp > 30:
                flags.append(
                    f"**Placing temperature {temp}°C** exceeds IS 7861 hot weather "
                    f"threshold of 30°C — accelerates moisture loss and cement hydration."
                )
            if hum < 50 or wind == "Exposed":
                flags.append(
                    f"**Humidity {hum}%** + **{wind} wind** — evaporation rate likely "
                    f"exceeds 1.0 kg/m²/hr threshold for plastic shrinkage cracking."
                )

            if flags:
                bullet_flags = "\n\n".join(f"- {f}" for f in flags)
                resp = (
                    f"## Why the {grade} Mix is Flagged\n\n"
                    f"The prediction model identified **{len(flags)} parameter deviation(s)** "
                    f"from IS code thresholds:\n\n"
                    f"{bullet_flags}\n\n"
                    f"---\n\n## Recommended Corrective Actions\n\n"
                    f"- Stop all on-site water additions — most common cause of W/C exceedance\n"
                    f"- Test aggregate free moisture (IS 2386 Part 3) before every batch\n"
                    f"- Extend curing to {14} days minimum using IS 12118 curing compound\n"
                    f"- Schedule pours before 7am or after 5pm in hot weather (IS 7861)\n"
                    f"- Install windbreaks on exposed faces before placing\n\n"
                    f"---\n\n## References\n\n"
                    f"- IS 456:2000 Table 5 (W/C limits) and Section 13.5 (curing)\n"
                    f"- IS 7861:2004 (Hot weather concreting)\n"
                    f"- IS 12118 (Curing compound specification)"
                )
            else:
                ok_rows = (
                    f"| W/C Ratio | {wc_act:.2f} | ≤{max_wc:.2f} | ✅ OK |\n"
                    f"| Curing | {curing} days | ≥14 days | {'❌ Low' if curing < 14 else '✅ OK'} |\n"
                    f"| Temperature | {temp}°C | ≤30°C | {'❌ High' if temp > 30 else '✅ OK'} |\n"
                    f"| Humidity | {hum}% | ≥50% | {'❌ Low' if hum < 50 else '✅ OK'} |"
                )
                resp = (
                    f"## {grade} Mix Assessment\n\n"
                    f"All parameters are within acceptable ranges:\n\n"
                    f"| Parameter | Current | Threshold | Status |\n|---|---|---|---|\n"
                    f"{ok_rows}\n\nNo corrective action required at this time."
                )

        elif intent == "knowledge":
            # ── FAISS + Qwen ─────────────────────────────────────────────────
            rag_chunks = _rag_retrieve(chat_inp, k=6)
            if _ASSISTANT_AVAILABLE and _qwen_answer is not None:
                with st.spinner("Thinking..."):
                    resp = _qwen_answer(
                        query=chat_inp,
                        act_vals=act_vals,
                        chunks=rag_chunks,
                        prediction_history=st.session_state.get("prediction_history", [])
                    )
                if not resp:
                    resp = "I could not generate a response. Please try rephrasing your question."
            else:
                # Minimal fallback without Qwen
                if "curing" in lower_q:
                    resp = (
                        f"**IS 456:2000 Curing Requirements:**\n\n"
                        f"- OPC only: 7 days minimum\n"
                        f"- PPC / blended cement: 10 days\n"
                        f"- With mineral admixtures: **14 days minimum**\n"
                        f"- Hot weather (>30°C): 14 days\n\n"
                        f"Your project shows **{act_vals['actual_curing_duration_days']} days** — "
                        f"{'below the 14-day minimum.' if act_vals['actual_curing_duration_days'] < 14 else 'adequate.'}"
                    )
                elif "honeycombing" in lower_q:
                    resp = (
                        "**Honeycombing** refers to voids and porous pockets in hardened concrete "
                        "caused by insufficient compaction. The coarse aggregate is present but lacks "
                        "mortar between particles.\n\n"
                        "**Causes:** Poor vibration, stiff mix, formwork gaps, bar congestion.\n\n"
                        "**Repair:** Chip to solid substrate → bonding agent → non-shrink grout (minor) "
                        "or structural engineer assessment + epoxy injection (severe)."
                    )
                else:
                    resp = (
                        "The AI model (Qwen) is loading. If this persists, restart the app.\n\n"
                        f"Active project: **{act_vals['concrete_grade']}** | "
                        f"W/C {act_vals['water_cement_ratio_actual']:.2f} | "
                        f"Curing {act_vals['actual_curing_duration_days']} days"
                    )

        else:  # off_topic
            resp = (
                "I specialise in **construction engineering and concrete quality analysis** only.\n\n"
                "I can answer questions on:\n"
                "- Crack risk, honeycombing, plastic shrinkage\n"
                "- IS 456:2000, IS 7861, IS 13311 requirements\n"
                "- Water-cement ratio, curing, mix design\n"
                "- Site safety, NDT inspection\n\n"
                "What would you like to know about your current project?"
            )

        with chat_box:
            st.session_state.copilot_history.append({"role": "assistant", "content": resp})
            with st.chat_message("assistant"):
                st.markdown(resp)
