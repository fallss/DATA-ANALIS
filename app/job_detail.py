import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from sklearn.preprocessing import MinMaxScaler
import requests

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="Ranked Talent Dashboard", layout="wide")
st.title("🎯 Ranked Talent Dashboard — Talent Match Intelligence")
st.markdown("Upload CSV atau ambil data dari Supabase. Hit `Run Analysis` untuk menghitung match rate, melihat ranked list, dan visualisasi.")

# ------------------------------
# UTILS: LLM (OpenRouter)
# ------------------------------
OPENROUTER_API_KEY = "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"  # optional
def generate_llm_response(prompt: str, model: str = "openai/gpt-4o-mini"):
    if not OPENROUTER_API_KEY:
        return "LLM API key not set — set OPENROUTER_API_KEY to enable automatic insights."
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert HR analyst who summarizes why employees are top performers."},
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ------------------------------
# SIDEBAR: data source & params
# ------------------------------
st.sidebar.header("Data Source & Analysis Settings")
data_source = st.sidebar.radio("Data source", ["Upload CSV", "Supabase"])
uploaded = None
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload combined_employee_data.csv (hasil Step 1)", type=["csv","xlsx"])
else:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    st.sidebar.markdown("Supabase connection from env vars.")
    st.sidebar.caption("Set SUPABASE_URL and SUPABASE_KEY in environment.")
st.sidebar.markdown("---")
st.sidebar.header("Success Formula Weights (tweakable)")
w_comp = st.sidebar.slider("Competency weight", 0.0, 1.0, 0.35, 0.05)
w_iq   = st.sidebar.slider("IQ weight", 0.0, 1.0, 0.25, 0.05)
w_exp  = st.sidebar.slider("Experience weight", 0.0, 1.0, 0.15, 0.05)
w_lead = st.sidebar.slider("Leadership weight", 0.0, 1.0, 0.15, 0.05)
w_other= st.sidebar.slider("Other numeric weight", 0.0, 1.0, 0.10, 0.05)
# Normalize weights
total_w = w_comp + w_iq + w_exp + w_lead + w_other
w_comp/=total_w; w_iq/=total_w; w_exp/=total_w; w_lead/=total_w; w_other/=total_w

st.sidebar.markdown(f"**Normalized weights:** Competency {w_comp:.2f}, IQ {w_iq:.2f}, Exp {w_exp:.2f}, Leadership {w_lead:.2f}, Other {w_other:.2f}")
st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶️ Run Analysis")

# ------------------------------
# Load data
# ------------------------------
df = pd.DataFrame()
load_error = None

if data_source == "Upload CSV" and uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        load_error = f"Failed load: {e}"

if data_source == "Supabase":
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            resp = supabase.table("employee_analysis").select("*").execute()
            if resp.error:
                load_error = f"Supabase error: {resp.error.message if hasattr(resp.error,'message') else resp.error}"
            else:
                df = pd.DataFrame(resp.data)
        except Exception as e:
            load_error = f"Supabase connection failed: {e}"
    else:
        load_error = "SUPABASE_URL/SUPABASE_KEY not set in environment."

if load_error:
    st.sidebar.error(load_error)

# show preview
if not df.empty:
    st.sidebar.success(f"Loaded {len(df)} rows — preview shown below")
    st.dataframe(df.head())

# ------------------------------
# Helper: choose numeric feature columns for scoring
# ------------------------------
def pick_numeric_columns(df):
    # candidate numeric columns commonly present in case data
    candidates = ["avg_competency_score","iq","years_experience","leadership_score","communication_score","cognitive_score"]
    existing = [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    # fallback: take top N numeric columns
    if len(existing) < 3:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        existing = nums[:6]
    return existing

# ------------------------------
# Main analysis triggered by button
# ------------------------------
if run_btn:
    if df.empty:
        st.error("No data loaded. Upload CSV or connect to Supabase first.")
        st.stop()

    # ensure employee_id and name fields exist
    id_col = next((c for c in ["employee_id","id","emp_id"] if c in df.columns), None)
    name_col = next((c for c in ["fullname","name","employee_name","nama"] if c in df.columns), None)
    if id_col is None:
        st.error("No employee id column found (expected employee_id). Add/rename that column.")
        st.stop()
    if name_col is None:
        st.warning("No obvious name column found — will use employee_id as label.")
        df["__display_name"] = df[id_col].astype(str)
        name_col = "__display_name"

    # pick numeric features for scoring and for top TGVs
    numeric_cols = pick_numeric_columns(df)
    st.sidebar.markdown(f"Numeric features used for scoring: {numeric_cols}")

    # prepare data for scoring: fill na, scale 0-1
    scaler = MinMaxScaler()
    numeric_df = df[numeric_cols].fillna(0).astype(float)
    try:
        scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_cols, index=df.index)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()

    # Build final match score using weights (map weights to available features)
    # We'll define groups: competency -> avg_competency_score, iq -> iq, exp -> years_experience, leadership->leadership_score, other->average of remaining numeric cols
    comp_val = scaled["avg_competency_score"] if "avg_competency_score" in scaled.columns else pd.Series(0, index=scaled.index)
    iq_val   = scaled["iq"] if "iq" in scaled.columns else pd.Series(0, index=scaled.index)
    exp_val  = scaled["years_experience"] if "years_experience" in scaled.columns else pd.Series(0, index=scaled.index)
    lead_val = scaled["leadership_score"] if "leadership_score" in scaled.columns else pd.Series(0, index=scaled.index)

    other_cols = [c for c in numeric_cols if c not in ["avg_competency_score","iq","years_experience","leadership_score"]]
    if other_cols:
        other_val = scaled[other_cols].mean(axis=1)
    else:
        other_val = pd.Series(0, index=scaled.index)

    final_score = (w_comp * comp_val + w_iq * iq_val + w_exp * exp_val + w_lead * lead_val + w_other * other_val)
    df["_final_score_raw"] = final_score
    df["final_match_rate"] = (final_score * 100).round(2)

    # Top TGVs/TVs per employee: top 3 numeric features where employee has highest scaled value
    def top_features_for_row(idx):
        row = scaled.loc[idx]
        topk = row.sort_values(ascending=False).head(3)
        return "; ".join([f"{col} ({row[col]:.2f})" for col in topk.index])

    df["top_TGVs"] = [top_features_for_row(i) for i in scaled.index]

    # strengths = features above global mean, gaps = below global mean
    global_mean = scaled.mean()
    def strengths_gaps(idx):
        row = scaled.loc[idx]
        strengths = row[row > global_mean].sort_values(ascending=False).index.tolist()
        gaps = row[row < global_mean].sort_values().index.tolist()
        return "; ".join(strengths[:5]), "; ".join(gaps[:5])

    sg = [strengths_gaps(i) for i in scaled.index]
    df["strengths"], df["gaps"] = zip(*sg)

    # ranked list
    ranked = df.sort_values("_final_score_raw", ascending=False).reset_index(drop=True)
    display_cols = [id_col, name_col, "final_match_rate", "top_TGVs", "strengths", "gaps"] + numeric_cols
    display_cols = [c for c in display_cols if c in ranked.columns]
    ranked_display = ranked[display_cols].copy()
    ranked_display.rename(columns={id_col:"employee_id", name_col:"name"}, inplace=True)

    st.markdown("## 🏅 Ranked Talent List")
    st.dataframe(ranked_display, use_container_width=True)

    # Download CSV button
    csv = ranked_display.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Ranked CSV", csv, "ranked_talent_list.csv", "text/csv")

    # ------------------------------
    # VISUALS
    # ------------------------------
    st.markdown("## 📈 Visualizations")

    # Match rate distribution
    fig = px.histogram(ranked, x="final_match_rate", nbins=20, title="Final Match Rate Distribution", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

    # Top strengths aggregated: count how often feature appears in top_TGVs across top N
    top_n = st.slider("Top N employees for aggregated strengths/gaps", 5, min(50, len(ranked)), 10)
    top_subset = ranked.head(top_n)
    # parse top_TGVs
    all_top_feats = []
    for s in top_subset["top_TGVs"].fillna(""):
        parts = [p.split(" (")[0] for p in s.split(";") if p.strip()]
        all_top_feats += parts
    feat_counts = pd.Series(all_top_feats).value_counts().rename_axis("feature").reset_index(name="count")
    fig_feats = px.bar(feat_counts.head(20), x="feature", y="count", title=f"Top features in top {top_n} employees")
    st.plotly_chart(fig_feats, use_container_width=True)

    # Radar: benchmark vs selected candidate
    st.markdown("### 🕸 Benchmark vs Candidate (Radar)")
    if numeric_cols:
        candidate_idx = st.selectbox("Select candidate (by employee id):", ranked_display["employee_id"].tolist())
        candidate_row = ranked[ranked[id_col]==candidate_idx].iloc[0]
        benchmark = scaled.mean().to_dict()
        candidate_vals = scaled.loc[ranked.index[ranked[id_col]==candidate_idx][0]].to_dict() if candidate_idx in ranked[id_col].values else scaled.iloc[0].to_dict()
        categories = numeric_cols
        bench_vals = [benchmark.get(c,0) for c in categories]
        cand_vals = [candidate_vals.get(c,0) for c in categories]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=bench_vals, theta=categories, fill='toself', name='Benchmark (mean)'))
        fig_radar.add_trace(go.Scatterpolar(r=cand_vals, theta=categories, fill='toself', name=str(candidate_idx)))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Heatmap correlation of numeric features
    st.markdown("### 🌡 Correlation Heatmap (Numeric features)")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig_heat = px.imshow(corr, text_auto=True, title="Correlation matrix")
        st.plotly_chart(fig_heat, use_container_width=True)

    # Bar of a few top metrics for the top K candidates
    st.markdown("### 📊 Top-K Candidate Metrics")
    k = st.slider("Show top K candidates", 3, min(50,len(ranked)), 10)
    topk_df = ranked.head(k)
    metrics = numeric_cols[:5]
    fig_bar = go.Figure()
    for m in metrics:
        if m in topk_df.columns:
            fig_bar.add_trace(go.Bar(name=m, x=topk_df[name_col].astype(str), y=topk_df[m]))
    fig_bar.update_layout(barmode='group', xaxis_tickangle=-45, title=f"Top {k} candidates metrics")
    st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# 🧠 AI Summary Insights (Fixed)
# ============================================================
st.markdown("## 🧠 Summary Insights (AI-generated)")

if st.button("🔎 Generate AI Insights for Top 5"):
    top5 = ranked.head(5)
    brief_lines = [
        f"{r.get(name_col, 'N/A')} (ID: {r[id_col]}) — Score: {r['final_match_rate']}, "
        f"Top: {r['top_TGVs']}, Strengths: {r['strengths']}"
        for _, r in top5.iterrows()
    ]
    prompt = (
        "You are an expert HR data analyst. "
        "Analyze the following top 5 employees based on their performance scores. "
        "Write 4 paragraphs covering: (1) overall performance patterns, "
        "(2) key skills or attributes driving success, "
        "(3) leadership or development insights, and "
        "(4) recommendations for HR strategy.\n\n"
        "Top Employees:\n" + "\n".join(brief_lines)
    )

    with st.spinner("🧠 Generating AI insights... please wait 10–20 seconds..."):
        try:
            insight = generate_llm_response(prompt)
            if insight.strip():
                st.success("✅ AI Insights generated successfully!")
                st.markdown(insight)
            else:
                st.warning("⚠️ No insights returned from AI. Try again or check your API key.")
        except Exception as e:
            st.error(f"❌ AI generation failed: {e}")

