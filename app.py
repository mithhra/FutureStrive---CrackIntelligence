import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import altair as alt
import datetime
from pathlib import Path

# ─── RAG Engine ───────────────────────────────────────────────────────────────
RAG_INDEX_FILE = Path("vector_store/index.faiss")
RAG_META_FILE  = Path("vector_store/chunk_metadata.json")

@st.cache_resource(show_spinner=False)
def _load_rag_engine():
    try:
        import faiss, json
        from sentence_transformers import SentenceTransformer
        if not RAG_INDEX_FILE.exists() or not RAG_META_FILE.exists():
            return None, None, None
        index = faiss.read_index(str(RAG_INDEX_FILE))
        with open(RAG_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return index, meta, model
    except Exception:
        return None, None, None

def _rag_retrieve(query: str, k: int = 6) -> list:
    index, meta, model = _load_rag_engine()
    if index is None or model is None:
        return []
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(meta):
            chunk = dict(meta[idx]); chunk["_score"] = float(dist)
            results.append(chunk)
    return results

# ─── Intent Router + Qwen LLM ─────────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_assistant():
    try:
        from knowledge_pipeline.intent_router import classify_query
        from knowledge_pipeline.llm_engine   import qwen_answer, qwen_crack_image_report
        return classify_query, qwen_answer, qwen_crack_image_report, True
    except Exception:
        return None, None, None, False

_classify_query, _qwen_answer, _qwen_image_report, _ASSISTANT_AVAILABLE = _load_assistant()
if not _ASSISTANT_AVAILABLE:
    def _classify_query(q, has_pending_image=False): return "knowledge"
    def _qwen_answer(query, act_vals, chunks, prediction_history=None, active_module="crack"): return None
    def _qwen_image_report(det, user_query="", chunks=None): return None

# ─── Reactive Crack Detector ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_crack_detector():
    try:
        from crack_detector import run_crack_detection, detection_to_summary_text
        return run_crack_detection, detection_to_summary_text, True
    except Exception:
        return None, None, False

_run_detection, _det_to_text, _DETECTOR_AVAILABLE = _load_crack_detector()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FutureStrive Construction Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family:'Inter',sans-serif !important; background:#FFFFFF !important; color:#1E293B !important; }
    .stMarkdown,.stMarkdown p,.stMarkdown span,.stMarkdown div,.stText,label { color:#1E293B !important; }
    h1,h2,h3,h4,h5,h6 { color:#0F172A !important; font-family:'Inter',sans-serif !important; font-weight:700 !important; letter-spacing:-0.02em !important; }
    div[data-testid="stMetricValue"] { color:#0F172A !important; font-weight:800 !important; }
    div[data-testid="stMetricLabel"] { color:#475569 !important; font-weight:600 !important; }

    [data-testid="stSidebar"] { background-color:#0F172A !important; border-right:1px solid #1E293B; }
    [data-testid="stSidebar"] * { color:#FFFFFF !important; }
    [data-testid="stSidebar"] p { color:#94A3B8 !important; }
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] { display:flex; flex-direction:column; gap:6px; }
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label {
        background-color:transparent !important; border:none !important;
        border-radius:6px !important; padding:10px 14px !important;
        cursor:pointer !important; font-size:13px !important; font-weight:500 !important; color:#94A3B8 !important;
    }
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label:hover { background-color:#1E293B !important; color:#FFFFFF !important; }
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] label[data-checked="true"] {
        background-color:#2563EB !important; color:#FFFFFF !important; font-weight:700 !important;
    }

    .saas-card { background:#F8FAFC !important; border:1px solid #E2E8F0 !important; border-radius:8px; padding:24px; margin-bottom:20px; }
    .saas-card * { color:#1E293B !important; }
    .result-card { background:#F0FDF4 !important; border:1px solid #BBF7D0 !important; border-radius:8px; padding:20px; margin-bottom:16px; }
    .result-card * { color:#1E293B !important; }
    .module-card {
        background:#F8FAFC; border:2px solid #E2E8F0; border-radius:12px;
        padding:32px 24px; text-align:center; cursor:pointer;
        transition:all 0.2s ease; margin-bottom:12px;
    }
    .module-card:hover { border-color:#2563EB; background:#EFF6FF; }
    .module-card h3 { color:#0F172A !important; font-size:20px; margin:12px 0 8px; }
    .module-card p { color:#64748B !important; font-size:14px; margin:0; }
    .copilot-prompt-card { background:#F1F5F9 !important; border:1px solid #E2E8F0 !important; border-radius:8px; padding:18px; margin-bottom:20px; }
    .copilot-prompt-card * { color:#1E293B !important; }
    .stButton > button { background:transparent !important; color:#1E293B !important; border:1px solid #CBD5E1 !important; box-shadow:none !important; transition:background 0.15s ease; }
    .stButton > button:hover { background:#E2E8F0 !important; border-color:#94A3B8 !important; }
    /* Chat input — light blue with visible dark border */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputContainer"] textarea,
    [data-baseweb="textarea"] textarea {
        background-color:#EFF6FF !important;
        color:#1E293B !important;
        caret-color:#2563EB !important;
        border:1.5px solid #334155 !important;
        border-radius:6px !important;
    }
    [data-testid="stChatInputContainer"],
    [data-testid="stBottom"] > div {
        background-color:#EFF6FF !important;
        border:2px solid #334155 !important;
        border-radius:8px !important;
        box-shadow:0 2px 8px rgba(37,99,235,0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = "Home"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "active_crack" not in st.session_state:
    st.session_state.active_crack = {
        "concrete_grade": "M35", "water_cement_ratio_design": 0.40,
        "water_cement_ratio_actual": 0.45, "actual_curing_duration_days": 8,
        "pour_temp": 34, "humidity": 40, "wind_exposure": "Normal",
    }
if "active_defect" not in st.session_state:
    st.session_state.active_defect = {
        "subcontractor_class": "Class B", "past_defect_rate_per_floor": 3.0,
        "skill_ratio": 0.70, "qc_hold_point_compliance_pct": 0.80,
        "spi": 1.00, "non_productive_days": 0, "approved_supplier": 1,
        "test_certificate_status": "Pass", "construction_stage": "Superstructure",
        "third_party_inspection": 0,
    }
if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = [{
        "role": "assistant",
        "content": (
            "## Welcome to the Construction Intelligence Assistant\n\n"
            "I am your AI advisor backed by IS codes, CPWD manuals, and live project data.\n\n"
            "**Crack Intelligence (Predictive):** Ask about crack probability, W/C ratio, curing, honeycombing\n\n"
            "**Defect Volume:** Ask about defect count, QC compliance, subcontractor quality, root cause\n\n"
            "**Crack Detection (Reactive):** Upload a site photo — I will run the detection model and generate an engineering report\n\n"
            "**Try:** *Why is the defect count high?* or *What is the IS 456 curing requirement?* or upload a crack photo below ⬇️"
        )
    }]
if "last_image_detection" not in st.session_state:
    st.session_state.last_image_detection = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
if "sidebar_nav" not in st.session_state:
    st.session_state.sidebar_nav = "Home"

def _on_sidebar_change():
    """Only called when user actually clicks a sidebar option."""
    st.session_state.nav_selection = st.session_state.sidebar_nav

with st.sidebar:
    st.markdown("## FutureStrive")
    st.markdown("**Construction Intelligence Platform**")
    st.markdown("---")
    # Sync sidebar indicator when user is on a sidebar-listed page
    _sidebar_options = ["Home", "Prediction History", "AI Assistant"]
    if st.session_state.nav_selection in _sidebar_options:
        st.session_state.sidebar_nav = st.session_state.nav_selection
    st.radio(
        "Navigation",
        _sidebar_options,
        key="sidebar_nav",
        on_change=_on_sidebar_change,
        label_visibility="collapsed"
    )

nav = st.session_state.nav_selection

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "Home":
    st.title("Construction Intelligence Platform")
    st.markdown("Select a module to begin prediction and analysis.")
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="module-card">
            <h3>Crack Intelligence</h3>
            <p>Predict crack occurrence probability, severity, and type from 21 pour parameters.
            Understand root causes with inline SHAP analysis and get corrective recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Crack Intelligence", use_container_width=True, key="goto_crack"):
            st.session_state.nav_selection = "Crack Intelligence"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="module-card">
            <h3>Defect Volume Intelligence</h3>
            <p>Predict defect count per floor, defect type distribution, severity grade, and root cause
            from trade, material, site, and historical QA/QC data.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Defect Volume Intelligence", use_container_width=True, key="goto_defect"):
            st.session_state.nav_selection = "Defect Volume"
            st.rerun()

    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    col3.metric("Total Predictions Run", len(st.session_state.prediction_history))
    crack_hist = [h for h in st.session_state.prediction_history if h.get("module") == "crack"]
    defect_hist = [h for h in st.session_state.prediction_history if h.get("module") == "defect"]
    col4.metric("Crack Predictions", len(crack_hist))
    col5.metric("Defect Predictions", len(defect_hist))


# ═══════════════════════════════════════════════════════════════════════════════
# CRACK INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Crack Intelligence":
    if st.button("← Back to Home", key="crack_back"):
        st.session_state.nav_selection = "Home"
        st.rerun()
    st.title("Crack Intelligence")
    st.markdown("Configure concrete pour parameters, run crack risk prediction, and review the analysis dashboard.")

    col_form, col_results = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.subheader("Pour Parameters")

        with st.expander("Concrete Mix Properties", expanded=True):
            grade    = st.selectbox("Concrete Grade", ["M25","M30","M35","M40"], index=2, key="c_grade")
            cem_type = st.selectbox("Cement Type", ["OPC 43 Grade","OPC 53 Grade","PPC","PSC"], index=1, key="c_cemtype")
            adm_type = st.selectbox("Admixture", ["No Admixture","Naphthalene-based superplasticiser","Polycarboxylate-based superplasticiser"], key="c_adm")
            wc_des   = st.slider("W/C Ratio (Design)", 0.35, 0.60, 0.40, 0.01, key="c_wcdes")
            wc_act   = st.slider("W/C Ratio (Actual)", 0.35, 0.65, 0.45, 0.01, key="c_wcact")
            wc_tol   = st.number_input("W/C Tolerance Spec", value=0.45, step=0.01, key="c_wctol")
            slump    = st.number_input("Target Slump (mm)", value=100, key="c_slump")
            agg_size = st.selectbox("Max Aggregate Size (mm)", [10, 20], index=1, key="c_agg")

        with st.expander("Curing & Execution", expanded=False):
            cur_meth = st.selectbox("Curing Method", ["Ponding","Sprinkling","Wet burlap curing","Curing compound"], key="c_curmeth")
            cur_plan = st.number_input("Planned Curing (days)", value=14, key="c_curplan")
            cur_act  = st.slider("Actual Curing (days)", 1, 28, 8, key="c_curact")
            cur_spec = st.number_input("Spec Min Curing (days)", value=14, key="c_curspec")
            pour_mo  = st.selectbox("Pour Month", ["January","February","March","April","May","June","July","August","September","October","November","December"], key="c_pourmo")
            chk      = st.slider("Checklist Sign-off Ratio", 0.0, 1.0, 1.0, key="c_chk")
            temp     = st.slider("Placing Temperature (C)", 10, 50, 34, key="c_temp")
            hum      = st.slider("Relative Humidity (%)", 10, 100, 40, key="c_hum")

        with st.expander("Site & Environment", expanded=False):
            wind       = st.selectbox("Wind Exposure", ["Sheltered","Normal","Exposed"], index=1, key="c_wind")
            shrink_risk= st.selectbox("Shrinkage Risk Season", ["LOW","MEDIUM","HIGH"], index=1, key="c_shrink")
            site_env   = st.selectbox("Site Environment", ["Inland","Coastal","Industrial"], key="c_env")
            city       = st.selectbox("City", ["Mumbai","Bangalore","Hyderabad","Delhi","Chennai"], key="c_city")
            access     = st.selectbox("Accessibility", ["Open","Semi-enclosed","Enclosed"], key="c_access")
            tier       = st.selectbox("Project Tier", ["Class A","Class B","Class C"], key="c_tier")
            sim_elems  = st.number_input("Similar Elements Count", value=10, key="c_simelems")

        run_crack = st.button("Run Crack Prediction", use_container_width=True, key="run_crack")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        if run_crack:
            feat_dict = {
                "concrete_grade": grade, "water_cement_ratio_design": wc_des,
                "water_cement_ratio_actual": wc_act, "cement_type": cem_type,
                "admixture_type": adm_type, "target_slump_mm": slump,
                "max_aggregate_size_mm": agg_size, "planned_pour_month": pour_mo,
                "curing_method": cur_meth, "planned_curing_duration_days": cur_plan,
                "actual_curing_duration_days": cur_act, "spec_min_curing_days": cur_spec,
                "wc_ratio_tolerance_spec": wc_tol, "pre_pour_checklist_signed_off_ratio": chk,
                "shrinkage_risk_season": shrink_risk, "wind_exposure_category": wind,
                "site_environment": site_env, "accessibility": access,
                "city": city, "project_tier": tier, "count_similar_elements": sim_elems,
            }
            features_df = pd.DataFrame([feat_dict])

            try:
                occ_model  = joblib.load("crack_occurrence_model.joblib")
                sev_model  = joblib.load("crack_severity_model.joblib")
                type_model = joblib.load("crack_type_model.joblib")
                occ_prob   = float(occ_model.predict_proba(features_df)[0][1])
                sev_label  = str(sev_model.predict(features_df)[0])
                type_label = str(type_model.predict(features_df)[0])
                if occ_prob < 0.25:
                    sev_label = "None (Low Risk)"; type_label = "None (Low Risk)"
                model_ok = True
            except Exception as e:
                st.error(f"Model Error: {e}")
                model_ok = False
                occ_prob = 0.0; sev_label = "Error"; type_label = "Error"

            # Update session state
            st.session_state.active_crack = {
                "concrete_grade": grade, "water_cement_ratio_design": wc_des,
                "water_cement_ratio_actual": wc_act, "actual_curing_duration_days": cur_act,
                "pour_temp": temp, "humidity": hum, "wind_exposure": wind,
            }
            record = {
                "module": "crack", "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "grade": grade, "wc_design": wc_des, "wc_actual": wc_act,
                "curing": cur_act, "temp": temp, "humidity": hum, "wind": wind,
                "prob": occ_prob, "severity": sev_label, "type": type_label,
            }
            st.session_state.prediction_history.append(record)

            # ── Results Dashboard ──────────────────────────────────────────────
            risk_color = "#DC2626" if occ_prob > 0.6 else ("#F59E0B" if occ_prob > 0.35 else "#16A34A")
            risk_label = "HIGH RISK" if occ_prob > 0.6 else ("MEDIUM RISK" if occ_prob > 0.35 else "LOW RISK")

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            st.markdown(
                f"""
                <div style='display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap'>
                  <div style='flex:1;min-width:120px;background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center'>
                    <div style='font-size:12px;color:#64748B;font-weight:600;margin-bottom:4px'>CRACK PROBABILITY</div>
                    <div style='font-size:26px;font-weight:800;color:#0F172A'>{occ_prob*100:.1f}%</div>
                  </div>
                  <div style='flex:1;min-width:120px;background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center'>
                    <div style='font-size:12px;color:#64748B;font-weight:600;margin-bottom:4px'>SEVERITY</div>
                    <div style='font-size:22px;font-weight:800;color:#0F172A'>{sev_label}</div>
                  </div>
                  <div style='flex:1;min-width:120px;background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center'>
                    <div style='font-size:12px;color:#64748B;font-weight:600;margin-bottom:4px'>CRACK TYPE</div>
                    <div style='font-size:16px;font-weight:800;color:#0F172A;word-break:break-word'>{type_label}</div>
                  </div>
                </div>
                <div style='background:{risk_color};color:#fff;padding:10px 18px;
                border-radius:6px;font-weight:700;font-size:16px;margin:4px 0'>
                {risk_label} &mdash; Crack Probability {occ_prob*100:.1f}%</div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Inline SHAP Analysis ───────────────────────────────────────────
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.subheader("Risk Driver Analysis")
            wc_dev  = wc_act - wc_des
            wc_c    = max(wc_dev*2.5 + (0.5 if wc_act > 0.45 else 0.0), 0.05)
            cur_c   = max((14 - cur_act)*0.12 if cur_act < 14 else 0.02, 0.02)
            wind_c  = 0.25 if wind=="Exposed" else (0.08 if wind=="Normal" else 0.02)
            temp_c  = max((temp-30)*0.035 if temp > 30 else 0.02, 0.02)
            hum_c   = max((50-hum)*0.010 if hum < 50 else 0.02, 0.02)
            grade_c = 0.12 if grade in ["M40","M35"] else 0.04
            chk_c   = max((1-chk)*0.15, 0.01)
            raw = {
                "W/C Ratio Deviation": wc_c, "Curing Duration": cur_c,
                "Wind Exposure": wind_c, "Placing Temperature": temp_c,
                "Relative Humidity": hum_c, "Concrete Grade": grade_c,
                "Checklist Compliance": chk_c,
            }
            tot = sum(raw.values())
            df_sh = pd.DataFrame([
                {"Feature": k, "Contribution %": round(v/tot*100, 1), "Risk": "High" if v/tot > 0.25 else ("Medium" if v/tot > 0.12 else "Low")}
                for k, v in raw.items()
            ]).sort_values("Contribution %", ascending=False).reset_index(drop=True)
            df_sh["Rank"] = df_sh.index + 1

            chart = alt.Chart(df_sh).mark_bar().encode(
                x=alt.X("Contribution %:Q"),
                y=alt.Y("Feature:N", sort="-x"),
                color=alt.Color("Risk:N", scale=alt.Scale(
                    domain=["High","Medium","Low"],
                    range=["#DC2626","#F59E0B","#16A34A"]
                )),
                tooltip=["Feature","Contribution %","Risk"]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_sh[["Rank","Feature","Contribution %","Risk"]], hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Corrective Recommendations ─────────────────────────────────────
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.subheader("Corrective Recommendations")
            max_wc = {"M25":0.55,"M30":0.50,"M35":0.45,"M40":0.40}.get(grade, 0.45)
            recs = []
            if wc_act > wc_des + 0.01:
                recs.append(f"**W/C Ratio:** Stop on-site water additions. Test aggregate free moisture per IS 2386 Part 3. IS 456:2000 max for {grade} is **{max_wc:.2f}**.")
            if cur_act < 14:
                recs.append(f"**Curing:** Extend from {cur_act} to **14 days** minimum. Use IS 12118 curing compound on exposed surfaces. IS 456:2000 Section 13.5.")
            if temp > 30:
                recs.append("**Hot Weather:** Schedule pours before 7am or after 5pm. Pre-cool mixing water with ice. Ref: IS 7861:2004.")
            if hum < 50 or wind == "Exposed":
                recs.append("**Evaporation Risk:** Install windbreaks on exposed faces. Apply evaporation retarder before finishing. Evaporation rate likely >1.0 kg/m2/hr.")
            if chk < 0.85:
                recs.append("**Checklist Compliance:** Pre-pour checklist sign-off is low. Ensure QC engineer completes all hold points before concrete placement.")
            if not recs:
                recs.append("All parameters are within acceptable ranges. No corrective action required at this time.")
            for r in recs:
                st.markdown(f"- {r}")
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.info("Configure parameters on the left and click **Run Crack Prediction** to see the full analysis dashboard.")
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DEFECT VOLUME INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Defect Volume":
    if st.button("← Back to Home", key="defect_back"):
        st.session_state.nav_selection = "Home"
        st.rerun()
    st.title("Defect Volume Intelligence")
    st.markdown("Configure project, trade, and site parameters to predict defect count, type, severity, and root cause.")

    col_form, col_results = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.subheader("Project Parameters")

        with st.expander("Project & Trade Profile", expanded=True):
            proj_type   = st.selectbox("Project Type", ["Residential","Commercial","Industrial","Infrastructure"], key="d_proj")
            gfa         = st.selectbox("GFA (sqm)", [5000,10000,20000,40000,80000], index=2, key="d_gfa")
            floors      = st.selectbox("Total Floors", [5,10,15,20,30,40], index=2, key="d_floors")
            struct_sys  = st.selectbox("Structural System", ["RCC Frame","Shear Wall","Flat Slab","Steel Frame"], key="d_struct")
            subcon      = st.selectbox("Subcontractor Class", ["Class A","Class B","Class C"], index=1, key="d_subcon")
            past_dr     = st.slider("Past Defect Rate per Floor", 0.5, 10.0, 3.0, 0.1, key="d_pastdr")
            workforce   = st.number_input("Workforce Size", value=80, key="d_wf")
            skill_r     = st.slider("Skill Ratio", 0.30, 0.95, 0.70, 0.01, key="d_skill")
            eng_exp     = st.number_input("Site Engineer Experience (yrs)", value=5, key="d_engexp")

        with st.expander("Activity & Progress", expanded=False):
            stage       = st.selectbox("Construction Stage", ["Foundation","Substructure","Superstructure","MEP Rough-in","Finishes","Facade"], index=2, key="d_stage")
            spi_val     = st.slider("SPI (Schedule Performance Index)", 0.60, 1.20, 1.00, 0.01, key="d_spi")
            conc_act    = st.number_input("Concurrent Activities", value=4, key="d_concact")
            prior_rw    = st.selectbox("Prior Rework on Same Element", ["No","Yes"], key="d_priorrw")
            qc_comp     = st.slider("QC Hold Point Compliance (%)", 0.40, 1.00, 0.80, 0.01, key="d_qccomp")
            third_party = st.selectbox("Third-Party Inspection", ["No","Yes"], key="d_tp")

        with st.expander("Material & Site", expanded=False):
            mat_grade   = st.selectbox("Material Grade", ["M25","M30","M35","M40"], index=1, key="d_matgrade")
            appr_sup    = st.selectbox("Approved Supplier", ["Yes","No"], key="d_appsup")
            del_var     = st.number_input("Delivery Variance (days)", value=0, key="d_delvar")
            test_cert   = st.selectbox("Test Certificate Status", ["Pass","Fail","Pending"], key="d_testcert")
            stor_cond   = st.selectbox("Site Storage Condition", ["Good","Fair","Poor"], key="d_stor")
            non_prod    = st.number_input("Non-Productive Days", value=0, key="d_nonprod")

        with st.expander("Historical QA/QC", expanded=False):
            defects_td  = st.number_input("Defects Recorded to Date", value=20, key="d_deftd")
            def_rate_cur= st.slider("Defect Rate (current project)", 0.5, 10.0, 3.0, 0.1, key="d_defratecur")
            top_def_t   = st.selectbox("Top Defect Type", ["Concrete Defects","Waterproofing","MEP Installation","Tiling/Finishing","Structural"], key="d_topdef")
            def_close   = st.slider("Defect Closure Rate (%)", 0.30, 1.00, 0.70, 0.01, key="d_defclose")
            port_avg    = st.slider("Portfolio Avg Defect Rate", 1.5, 5.0, 3.0, 0.1, key="d_portavg")

        run_defect = st.button("Run Defect Prediction", use_container_width=True, key="run_defect")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        if run_defect:
            feat_dict = {
                "project_type": proj_type, "gfa_sqm": gfa, "total_floors": floors,
                "structural_system": struct_sys, "subcontractor_class": subcon,
                "past_defect_rate_per_floor": past_dr, "workforce_size": workforce,
                "skill_ratio": skill_r, "site_engineer_experience_yrs": eng_exp,
                "construction_stage": stage, "spi": spi_val,
                "concurrent_activities": conc_act,
                "prior_rework_same_element": 1 if prior_rw == "Yes" else 0,
                "qc_hold_point_compliance_pct": qc_comp,
                "third_party_inspection": 1 if third_party == "Yes" else 0,
                "material_grade": mat_grade,
                "approved_supplier": 1 if appr_sup == "Yes" else 0,
                "delivery_variance_days": del_var,
                "test_certificate_status": test_cert,
                "site_storage_condition": stor_cond,
                "non_productive_days": non_prod,
                "defects_recorded_to_date": defects_td,
                "defect_rate_current_project": def_rate_cur,
                "top_defect_type_1": top_def_t,
                "defect_closure_rate_pct": def_close,
                "portfolio_avg_defect_rate": port_avg,
            }
            features_df = pd.DataFrame([feat_dict])

            try:
                cnt_model  = joblib.load("defect_count_model.joblib")
                type_model = joblib.load("defect_type_model.joblib")
                sev_model  = joblib.load("defect_severity_model.joblib")
                rc_model   = joblib.load("defect_rootcause_model.joblib")

                pred_count   = int(round(float(cnt_model.predict(features_df)[0])))
                type_le      = type_model.label_encoder_
                sev_le       = sev_model.label_encoder_
                rc_le        = rc_model.label_encoder_
                pred_type    = type_le.inverse_transform(type_model.predict(features_df))[0]
                pred_sev     = sev_le.inverse_transform(sev_model.predict(features_df))[0]
                pred_rc      = rc_le.inverse_transform(rc_model.predict(features_df))[0]
                model_ok     = True
            except Exception as e:
                st.error(f"Model Error: {e}")
                model_ok   = False
                pred_count = 0; pred_type = "Error"; pred_sev = "Error"; pred_rc = "Error"

            # Update session state
            st.session_state.active_defect = {
                "subcontractor_class": subcon, "past_defect_rate_per_floor": past_dr,
                "skill_ratio": skill_r, "qc_hold_point_compliance_pct": qc_comp,
                "spi": spi_val, "non_productive_days": non_prod,
                "approved_supplier": 1 if appr_sup=="Yes" else 0,
                "test_certificate_status": test_cert,
                "construction_stage": stage,
                "third_party_inspection": 1 if third_party=="Yes" else 0,
                "last_defect_prediction": {
                    "count": pred_count, "type": pred_type,
                    "severity": pred_sev, "root_cause": pred_rc
                }
            }
            record = {
                "module": "defect", "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "subcon": subcon, "stage": stage, "spi": spi_val,
                "count": pred_count, "type": pred_type,
                "severity": pred_sev, "root_cause": pred_rc,
            }
            st.session_state.prediction_history.append(record)

            # Risk classification
            if pred_count >= 25 or pred_sev == "Critical":
                risk_color, risk_label = "#DC2626", "HIGH DEFECT RISK"
            elif pred_count >= 15 or pred_sev == "Major":
                risk_color, risk_label = "#F59E0B", "MEDIUM DEFECT RISK"
            else:
                risk_color, risk_label = "#16A34A", "LOW DEFECT RISK"

            # ── Results Dashboard ──────────────────────────────────────────────
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            st.markdown(
                f"""
                <div style='display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap'>
                  <div style='flex:1;min-width:110px;background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center'>
                    <div style='font-size:11px;color:#64748B;font-weight:600;margin-bottom:4px'>DEFECT COUNT / FLOOR</div>
                    <div style='font-size:28px;font-weight:800;color:#0F172A'>{pred_count}</div>
                  </div>
                  <div style='flex:1;min-width:110px;background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center'>
                    <div style='font-size:11px;color:#64748B;font-weight:600;margin-bottom:4px'>DOMINANT TYPE</div>
                    <div style='font-size:18px;font-weight:800;color:#0F172A;word-break:break-word'>{pred_type}</div>
                  </div>
                  <div style='flex:1;min-width:110px;background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center'>
                    <div style='font-size:11px;color:#64748B;font-weight:600;margin-bottom:4px'>SEVERITY GRADE</div>
                    <div style='font-size:18px;font-weight:800;color:#0F172A'>{pred_sev}</div>
                  </div>
                  <div style='flex:1;min-width:110px;background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center'>
                    <div style='font-size:11px;color:#64748B;font-weight:600;margin-bottom:4px'>ROOT CAUSE</div>
                    <div style='font-size:18px;font-weight:800;color:#0F172A;word-break:break-word'>{pred_rc}</div>
                  </div>
                </div>
                <div style='background:{risk_color};color:#fff;padding:10px 18px;
                border-radius:6px;font-weight:700;font-size:16px;margin:4px 0'>
                {risk_label} &mdash; {pred_count} defects/floor predicted ({pred_sev})</div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Inline Driver Analysis ─────────────────────────────────────────
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.subheader("Defect Driver Analysis")

            proc_c = (1 - qc_comp) * 10 + (0 if third_party == "Yes" else 3)
            mat_c  = (0 if appr_sup == "Yes" else 8) + (4 if test_cert == "Fail" else 0) + (1 if stor_cond == "Poor" else 0)
            sup_c  = max(0, (5 - eng_exp) * 0.8) + (1 - skill_r) * 5
            wea_c  = non_prod * 0.6
            sch_c  = max(0, (1/spi_val - 1) * 6) if spi_val > 0 else 0
            his_c  = max(0, past_dr * 0.5)

            raw_d = {
                "Process / QC Compliance": max(proc_c, 0.1),
                "Material Quality": max(mat_c, 0.1),
                "Supervision / Skill": max(sup_c, 0.1),
                "Weather / Non-Productive Days": max(wea_c, 0.1),
                "Schedule Pressure (SPI)": max(sch_c, 0.1),
                "Historical Defect Rate": max(his_c, 0.1),
            }
            tot_d = sum(raw_d.values())
            df_d = pd.DataFrame([
                {"Driver": k, "Contribution %": round(v/tot_d*100, 1),
                 "Risk": "High" if v/tot_d > 0.25 else ("Medium" if v/tot_d > 0.12 else "Low")}
                for k, v in raw_d.items()
            ]).sort_values("Contribution %", ascending=False).reset_index(drop=True)
            df_d["Rank"] = df_d.index + 1

            chart_d = alt.Chart(df_d).mark_bar().encode(
                x=alt.X("Contribution %:Q"),
                y=alt.Y("Driver:N", sort="-x"),
                color=alt.Color("Risk:N", scale=alt.Scale(
                    domain=["High","Medium","Low"],
                    range=["#DC2626","#F59E0B","#16A34A"]
                )),
                tooltip=["Driver","Contribution %","Risk"]
            ).properties(height=260)
            st.altair_chart(chart_d, use_container_width=True)
            st.dataframe(df_d[["Rank","Driver","Contribution %","Risk"]], hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Recommendations ────────────────────────────────────────────────
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.subheader("Improvement Recommendations")
            recs = []
            if qc_comp < 0.80:
                recs.append(f"**QC Compliance is {qc_comp:.0%}** — Increase QC hold points. Mandate sign-off before each pour. Target >90%.")
            if appr_sup == "No" or test_cert in ["Fail","Pending"]:
                recs.append("**Material Risk** — Switch to approved supplier. Do not use materials with failed or pending test certificates.")
            if skill_r < 0.60:
                recs.append(f"**Low Skill Ratio ({skill_r:.0%})** — Deploy additional skilled tradespeople. Skill ratio below 60% correlates directly with process defects.")
            if eng_exp < 4:
                recs.append(f"**Supervision** — Site engineer has {eng_exp} yrs experience. Assign a senior engineer as back-up QA reviewer for critical activities.")
            if spi_val < 0.90:
                recs.append(f"**Schedule Pressure (SPI={spi_val:.2f})** — Project is behind schedule. Avoid rushing activities. Review the programme and add float for QC activities.")
            if third_party == "No" and pred_count > 15:
                recs.append("**Third-Party Inspection** — At this defect rate, engage an independent QA inspector. Projects with TPI have 30-50% lower defect escape rates.")
            if not recs:
                recs.append("All key parameters are within acceptable ranges. Maintain current QC standards.")
            for r in recs:
                st.markdown(f"- {r}")
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.info("Configure parameters on the left and click **Run Defect Prediction** to see the full analysis dashboard.")
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Prediction History":
    st.title("Prediction History")

    if not st.session_state.prediction_history:
        st.info("No predictions yet. Run a prediction from Crack Intelligence or Defect Volume.")
    else:
        crack_hist  = [h for h in st.session_state.prediction_history if h.get("module") == "crack"]
        defect_hist = [h for h in st.session_state.prediction_history if h.get("module") == "defect"]

        tab1, tab2 = st.tabs(["Crack Intelligence", "Defect Volume"])

        with tab1:
            if crack_hist:
                df_c = pd.DataFrame(crack_hist)
                df_c["Probability %"] = (df_c["prob"] * 100).round(1)
                st.dataframe(
                    df_c[["timestamp","grade","wc_actual","curing","temp","humidity","Probability %","severity","type"]],
                    use_container_width=True, hide_index=True
                )
                if len(df_c) > 1:
                    line = alt.Chart(df_c.reset_index()).mark_line(point=True, color="#2563EB").encode(
                        x=alt.X("index:O", title="Prediction #"),
                        y=alt.Y("Probability %:Q", scale=alt.Scale(domain=[0,100])),
                        tooltip=["timestamp","grade","Probability %","severity"]
                    ).properties(height=280, title="Crack Probability Trend")
                    st.altair_chart(line, use_container_width=True)
            else:
                st.info("No crack predictions yet.")

        with tab2:
            if defect_hist:
                df_d = pd.DataFrame(defect_hist)
                st.dataframe(
                    df_d[["timestamp","subcon","stage","spi","count","type","severity","root_cause"]],
                    use_container_width=True, hide_index=True
                )
                if len(df_d) > 1:
                    bar = alt.Chart(df_d.reset_index()).mark_bar(color="#7C3AED").encode(
                        x=alt.X("index:O", title="Prediction #"),
                        y=alt.Y("count:Q", title="Defect Count / Floor"),
                        tooltip=["timestamp","subcon","count","severity","root_cause"]
                    ).properties(height=280, title="Defect Count Trend")
                    st.altair_chart(bar, use_container_width=True)
            else:
                st.info("No defect volume predictions yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT (Unified)
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "AI Assistant":
    st.title("AI Quality Assistant")
    st.markdown("Powered by **Qwen 2.5** · Intent-routed · Context-aware across Crack and Defect Volume modules")

    act_crack  = st.session_state.active_crack
    act_defect = st.session_state.active_defect

    # Context banner
    st.markdown(f"""
    <div style="background:#F1F5F9;border:1px solid #E2E8F0;padding:10px 16px;
    border-radius:6px;margin-bottom:16px;font-size:12.5px;color:#1E293B;">
    <strong>Crack Context:</strong>
    Grade: <code>{act_crack.get('concrete_grade','M35')}</code> ·
    W/C: <code>{act_crack.get('water_cement_ratio_actual',0.45):.2f}</code> ·
    Curing: <code>{act_crack.get('actual_curing_duration_days',8)} days</code>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>Defect Context:</strong>
    Subcon: <code>{act_defect.get('subcontractor_class','Class B')}</code> ·
    QC: <code>{act_defect.get('qc_hold_point_compliance_pct',0.80):.0%}</code> ·
    SPI: <code>{act_defect.get('spi',1.00):.2f}</code>
    </div>
    """, unsafe_allow_html=True)

    # Quick prompts
    st.markdown("<div class='copilot-prompt-card'>", unsafe_allow_html=True)
    st.markdown("##### Quick questions:")
    c1, c2, c3, c4 = st.columns(4)
    sh_query = ""
    with c1:
        if st.button("Why is mix flagged?",      use_container_width=True, key="q1"): sh_query = "Why is the current mix flagged?"
    with c2:
        if st.button("Why is defect count high?",use_container_width=True, key="q2"): sh_query = "Why is the defect count high?"
    with c3:
        if st.button("What is honeycombing?",    use_container_width=True, key="q3"): sh_query = "What is honeycombing and how do I fix it?"
    with c4:
        if st.button("Improve QC compliance",    use_container_width=True, key="q4"): sh_query = "How do I improve QC hold point compliance?"
    st.markdown("</div>", unsafe_allow_html=True)

    # ─── Reactive: Image Upload ─────────────────────────────────────────────────────
    with st.expander("📷  Upload Crack Photo — Reactive Detection", expanded=False):
        st.markdown(
            "<div style='font-size:13px;color:#475569;margin-bottom:10px;'>"
            "Upload a site photo of a crack surface. The detection model will analyse it and the "
            "AI assistant will generate a structured engineering report with IS code references."
            "</div>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="crack_img_upload",
            label_visibility="collapsed",
        )
        col_img_l, col_img_r = st.columns([1, 1], gap="medium")
        with col_img_l:
            if uploaded_file is not None:
                from PIL import Image as PILImage
                pil_img = PILImage.open(uploaded_file).convert("RGB")
                # Resize for display if too large
                max_w = 600
                if pil_img.width > max_w:
                    ratio = max_w / pil_img.width
                    pil_img = pil_img.resize((max_w, int(pil_img.height * ratio)), PILImage.LANCZOS)
                st.image(pil_img, caption="Uploaded image", use_container_width=True)

                analyse_btn = st.button(
                    "🔍 Run Crack Detection & Generate Report",
                    use_container_width=True,
                    key="analyse_img_btn",
                    type="primary",
                )

                if analyse_btn:
                    if not _DETECTOR_AVAILABLE:
                        st.error("Crack detector module not available. Check crack_detector.py.")
                    else:
                        with st.spinner("Running crack detection model..."):
                            det = _run_detection(pil_img)
                            st.session_state.last_image_detection = det

                        with col_img_r:
                            ann = det.get("annotated_image")
                            if ann is not None:
                                st.image(ann, caption="Detection overlay", use_container_width=True)

                        # Show model mode banner — makes errors visible
                        _mode = det.get("model_mode", "stub")
                        if _mode == "real":
                            st.success("✅ Live model — DeepLabV3+ (ResNet-101)")
                        elif _mode == "stub":
                            st.info("ℹ️ Demo mode — set USE_STUB=False and provide model path to use real model")
                        else:
                            st.warning(f"⚠️ Fell back to demo mode. Reason: `{_mode}`")

                        # Build user chat message
                        mode_note = " *(demo mode)*" if "stub" in _mode else ""
                        user_msg = (
                            f"📷 **Crack photo uploaded{mode_note}** — please analyse and generate an engineering report."
                        )

                        # Route through intent: image_crack_analysis
                        rag_chunks = _rag_retrieve("crack severity type IS 456 remediation", k=4)

                        if _ASSISTANT_AVAILABLE and _qwen_image_report is not None:
                            with st.spinner("Generating engineering report..."):
                                report = _qwen_image_report(
                                    det=det,
                                    user_query=user_msg,
                                    chunks=rag_chunks,
                                )
                        else:
                            # Fallback: import fallback directly from llm_engine
                            try:
                                from knowledge_pipeline.llm_engine import _fallback_image_report
                                report = _fallback_image_report(det)
                            except Exception:
                                report = "Detection complete. LLM report unavailable — check knowledge_pipeline setup."

                        # Append to chat history so it appears in the chat thread
                        st.session_state.copilot_history.append({
                            "role": "user", "content": user_msg
                        })
                        st.session_state.copilot_history.append({
                            "role": "assistant", "content": report
                        })
                        st.rerun()  # refresh chat display

        # Show previous detection result if available
        with col_img_r:
            last_det = st.session_state.last_image_detection
            if last_det and uploaded_file is None:
                ann = last_det.get("annotated_image")
                if ann is not None:
                    st.image(ann, caption="Last detection overlay", use_container_width=True)

    # Chat history
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.copilot_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    chat_inp = st.chat_input("Ask about crack risk, defect volume, IS 456, curing, QC compliance, subcontractor quality...")
    if sh_query:
        chat_inp = sh_query

    if chat_inp:
        st.session_state.copilot_history.append({"role": "user", "content": chat_inp})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(chat_inp)

        lower_q  = chat_inp.lower()
        has_img  = st.session_state.last_image_detection is not None and any(
            kw in lower_q for kw in [
                "analyse", "analyze", "image", "photo", "picture", "crack image",
                "detect", "uploaded", "report", "what crack", "what type"
            ]
        )
        intent   = _classify_query(chat_inp, has_pending_image=has_img)

        # ── Analytical handler ────────────────────────────────────────────────
        _NUM_PATTERN = re.compile(
            r"\b(less than|greater than|more than|above|below|under|at least|at most"
            r"|fewer than|equal to|<|>|<=|>=|==)\s+([\d.]+)\b",
            re.IGNORECASE
        )

        def _map_op(raw):
            r = raw.lower().strip()
            if r in ("<","less than","below","under","fewer than"): return "<"
            if r in (">","greater than","above","over","more than"): return ">"
            if r in ("<=","at most"): return "<="
            if r in (">=","at least"): return ">="
            if r in ("==","equal to"): return "=="
            return None

        def _crack_params_df():
            wmap = {"Sheltered":0,"Normal":1,"Exposed":2}
            gmap = {"M25":25,"M30":30,"M35":35,"M40":40}
            return pd.DataFrame([
                {"Parameter":"W/C Design",  "Value": act_crack.get("water_cement_ratio_design",0.40), "Unit":"-"},
                {"Parameter":"W/C Actual",  "Value": act_crack.get("water_cement_ratio_actual",0.45),  "Unit":"-"},
                {"Parameter":"Curing Days", "Value": act_crack.get("actual_curing_duration_days",8),   "Unit":"days"},
                {"Parameter":"Temperature", "Value": act_crack.get("pour_temp",30),                    "Unit":"C"},
                {"Parameter":"Humidity",    "Value": act_crack.get("humidity",50),                     "Unit":"%"},
                {"Parameter":"Wind Exposure","Value": wmap.get(act_crack.get("wind_exposure","Normal"),1),"Unit":"0-2"},
                {"Parameter":"Grade (MPa)", "Value": gmap.get(act_crack.get("concrete_grade","M35"),35),"Unit":"MPa"},
            ])

        def _defect_params_df():
            return pd.DataFrame([
                {"Parameter":"Past Defect Rate",   "Value": act_defect.get("past_defect_rate_per_floor",3.0), "Unit":"defects/floor"},
                {"Parameter":"Skill Ratio",         "Value": act_defect.get("skill_ratio",0.70),              "Unit":"%"},
                {"Parameter":"QC Compliance",       "Value": act_defect.get("qc_hold_point_compliance_pct",0.80),"Unit":"%"},
                {"Parameter":"SPI",                 "Value": act_defect.get("spi",1.00),                      "Unit":"-"},
                {"Parameter":"Non-Productive Days", "Value": act_defect.get("non_productive_days",0),         "Unit":"days"},
                {"Parameter":"Approved Supplier",   "Value": act_defect.get("approved_supplier",1),           "Unit":"0/1"},
            ])

        _pandas_resp = None
        num_match = _NUM_PATTERN.search(chat_inp)
        if num_match or intent == "analytical":
            if num_match:
                op_raw, thr_str = num_match.group(1), num_match.group(2)
                op  = _map_op(op_raw); thr = float(thr_str)
                is_defect = any(kw in lower_q for kw in ["defect","qc","compliance","spi","skill","subcon"])
                if op:
                    df_src = _defect_params_df() if is_defect else _crack_params_df()
                    filtered = df_src.query(f"Value {op} @thr")
                    label = "defect" if is_defect else "crack"
                    if filtered.empty:
                        _pandas_resp = f"No {label} parameters with value {op_raw} {thr}.\n\n```\n{df_src.to_string(index=False)}\n```"
                    else:
                        _pandas_resp = f"**{label.capitalize()} parameters where Value {op_raw} {thr}:**\n\n```\n{filtered.to_string(index=False)}\n```"

        # ── Determine active module context ───────────────────────────────────
        active_module = "defect" if intent == "defect_prediction" else "crack"
        act_vals = act_defect if active_module == "defect" else act_crack

        # ── Route to correct handler ──────────────────────────────────────────
        if intent == "greeting":
            resp = (
                "Hello! I'm your **Construction Intelligence Assistant** — unified across both modules.\n\n"
                "**Crack Intelligence** — Ask me about:\n"
                "- *Why is the W/C ratio flagged?* · *What curing duration is required?*\n"
                "- *What is honeycombing?* · *How do I prevent plastic shrinkage cracks?*\n\n"
                "**Defect Volume** — Ask me about:\n"
                "- *Why is the defect count high?* · *What is causing defects on this floor?*\n"
                "- *How do I improve QC compliance?* · *What does SPI mean?*\n\n"
                f"Active crack context: **{act_crack.get('concrete_grade','M35')}** | "
                f"W/C {act_crack.get('water_cement_ratio_actual',0.45):.2f} | "
                f"Curing {act_crack.get('actual_curing_duration_days',8)} days\n\n"
                f"Active defect context: **{act_defect.get('subcontractor_class','Class B')}** | "
                f"QC {act_defect.get('qc_hold_point_compliance_pct',0.80):.0%} | "
                f"SPI {act_defect.get('spi',1.00):.2f}"
            )

        elif _pandas_resp is not None:
            resp = _pandas_resp

        elif intent == "crack_prediction":
            wc_act = act_crack.get("water_cement_ratio_actual", 0.45)
            wc_des = act_crack.get("water_cement_ratio_design", 0.40)
            curing = act_crack.get("actual_curing_duration_days", 8)
            temp   = act_crack.get("pour_temp", 30)
            hum    = act_crack.get("humidity", 50)
            wind   = act_crack.get("wind_exposure", "Normal")
            grade  = act_crack.get("concrete_grade", "M35")
            max_wc = {"M25":0.55,"M30":0.50,"M35":0.45,"M40":0.40}.get(grade, 0.45)
            flags = []
            if wc_act > wc_des + 0.01:
                flags.append(f"**W/C {wc_act:.2f}** exceeds design **{wc_des:.2f}** (IS 456 max for {grade}: {max_wc:.2f})")
            if curing < 14:
                flags.append(f"**Curing {curing} days** — {14-curing} days below IS 456:2000 minimum of 14 days")
            if temp > 30:
                flags.append(f"**Temperature {temp}C** exceeds IS 7861 hot weather threshold")
            if hum < 50 or wind == "Exposed":
                flags.append(f"**Humidity {hum}% + {wind} wind** — evaporation risk exceeds 1.0 kg/m2/hr")
            if flags:
                resp = (f"## Crack Risk Analysis — {grade}\n\n"
                        f"**{len(flags)} parameter deviation(s)** from IS code thresholds:\n\n"
                        + "\n".join(f"- {f}" for f in flags)
                        + "\n\n---\n\n## Corrective Actions\n\n"
                        "- Stop on-site water additions — check aggregate free moisture (IS 2386 Part 3)\n"
                        "- Extend curing to 14 days minimum using IS 12118 curing compound\n"
                        "- Schedule pours before 7am or after 5pm in hot weather (IS 7861)\n"
                        "- Install windbreaks on exposed faces")
            else:
                resp = f"## {grade} Mix Assessment\n\nAll crack parameters are within acceptable IS 456:2000 ranges. No corrective action required."

        elif intent == "defect_prediction":
            subcon  = act_defect.get("subcontractor_class", "Class B")
            qc      = act_defect.get("qc_hold_point_compliance_pct", 0.80)
            spi_v   = act_defect.get("spi", 1.00)
            skill   = act_defect.get("skill_ratio", 0.70)
            tp      = act_defect.get("third_party_inspection", 0)
            pred    = act_defect.get("last_defect_prediction", {})
            flags   = []
            if qc < 0.80:
                flags.append(f"**QC compliance {qc:.0%}** — below 80% threshold. Each missed hold point increases defect escape rate.")
            if subcon == "Class C":
                flags.append("**Class C subcontractor** — historical defect rate 2-4x higher than Class A. Increase inspection frequency.")
            if spi_v < 0.90:
                flags.append(f"**SPI {spi_v:.2f}** — project is behind schedule. Rushed work directly increases defect rates.")
            if skill < 0.60:
                flags.append(f"**Skill ratio {skill:.0%}** — below 60%. Process defects increase with lower skilled workforce ratio.")
            if not tp:
                flags.append("**No third-party inspection** — projects without TPI see 30-50% higher defect escape rates.")
            count_str = f"Predicted {pred.get('count','?')} defects/floor ({pred.get('severity','?')} / {pred.get('type','?')} / Root: {pred.get('root_cause','?')}). " if pred else ""
            resp = (f"## Defect Volume Analysis\n\n{count_str}"
                    f"**{len(flags)} risk factor(s) identified:**\n\n"
                    + "\n".join(f"- {f}" for f in flags)
                    + "\n\n---\n\n## Recommendations\n\n"
                    "- Increase QC hold points — mandate sign-off before each activity proceeds\n"
                    "- Engage third-party inspection if defect count >15/floor\n"
                    "- Deploy additional skilled tradespeople and review the programme\n"
                    "- Conduct pre-activity toolbox talks to address process defects")

        elif intent == "image_crack_analysis":
            last_det = st.session_state.get("last_image_detection")
            if last_det is None:
                resp = (
                    "## 📷 No Image Detected\n\n"
                    "I don't have a crack photo to analyse yet. Please:\n\n"
                    "1. Expand the **'Upload Crack Photo — Reactive Detection'** panel above\n"
                    "2. Upload a site image (JPG / PNG)\n"
                    "3. Click **Run Crack Detection & Generate Report**\n\n"
                    "Once analysed, you can also ask me follow-up questions like:\n"
                    "- *What type of crack is this?*\n"
                    "- *What does the severity mean?*\n"
                    "- *What IS code applies to this crack?*"
                )
            else:
                rag_chunks = _rag_retrieve("crack severity remediation IS 456 epoxy injection", k=4)
                if _ASSISTANT_AVAILABLE and _qwen_image_report is not None:
                    with st.spinner("Re-analysing detection result..."):
                        resp = _qwen_image_report(
                            det=last_det,
                            user_query=chat_inp,
                            chunks=rag_chunks,
                        )
                else:
                    try:
                        from knowledge_pipeline.llm_engine import _fallback_image_report
                        resp = _fallback_image_report(last_det)
                    except Exception:
                        resp = "Report generation unavailable. Check knowledge_pipeline setup."
                if not resp:
                    resp = "Could not generate report. Please try rephrasing or re-uploading the image."

        elif intent == "knowledge":
            rag_chunks = _rag_retrieve(chat_inp, k=6)
            if _ASSISTANT_AVAILABLE and _qwen_answer is not None:
                with st.spinner("Thinking..."):
                    resp = _qwen_answer(
                        query=chat_inp, act_vals=act_vals, chunks=rag_chunks,
                        prediction_history=st.session_state.get("prediction_history", []),
                        active_module=active_module
                    )
                if not resp:
                    resp = "I could not generate a response. Please try rephrasing your question."
            else:
                resp = "The AI model (Qwen) is loading. If this persists, restart the app."

        else:  # off_topic
            resp = (
                "I specialise in **construction engineering** — crack intelligence and defect volume analysis.\n\n"
                "Ask me about crack risk, honeycombing, IS 456:2000, curing, defect count, "
                "QC compliance, subcontractor performance, or SPI."
            )

        with chat_box:
            st.session_state.copilot_history.append({"role": "assistant", "content": resp})
            with st.chat_message("assistant"):
                st.markdown(resp)
