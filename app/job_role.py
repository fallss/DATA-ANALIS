# ============================================================
# 🎯 Talent Match Intelligence System (Local Excel Version Only)
# ============================================================

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ============================================================
# CONFIGURATIONS
# ============================================================

st.set_page_config(page_title="Talent Match Intelligence", layout="wide")
st.title("🎯 Talent Match Intelligence System (Local Excel Version)")
st.markdown("---")

# ============================================================
# 🔮 LLM via OpenRouter
# ============================================================

OPENROUTER_API_KEY = "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"


def generate_llm_response(prompt: str, model: str = "openai/gpt-4o-mini") -> str:
    """Generate AI response from OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert HR talent intelligence assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"].strip()


# ============================================================
# 🧰 SIDEBAR INPUT
# ============================================================

with st.sidebar:
    st.header("Job Vacancy Configuration")
    role_name = st.text_input("Role Name", "Data Analyst")
    job_level = st.selectbox("Job Level", ["Junior", "Middle", "Senior", "Lead"])
    role_purpose = st.text_area(
        "Role Purpose",
        "Analyze business data and create actionable insights to support strategic decisions.",
    )

    st.subheader("📂 Upload Employee Data")
    uploaded_file = st.file_uploader("Upload Excel/CSV file", type=["csv", "xlsx"])
    max_candidates = st.slider("Max Candidates to Analyze", 3, 20, 10)

# ============================================================
# 📥 LOAD DATA
# ============================================================

df = pd.DataFrame()

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
    st.success(f"✅ Data berhasil dimuat! Jumlah baris: {len(df)}")
    st.dataframe(df.head(), use_container_width=True)

# ============================================================
# 📊 VISUALIZATION & ANALYSIS
# ============================================================

if not df.empty:
    if "rating" not in df.columns:
        st.error("Kolom 'rating' tidak ditemukan! Pastikan data memiliki kolom 'rating'.")
    else:
        top_candidates = df[df["rating"] == df["rating"].max()]
        st.markdown("### 👥 Kandidat dengan Rating Tertinggi")
        st.dataframe(top_candidates, use_container_width=True)

        # ============================================================
        # VISUALIZATION SECTION
        # ============================================================
        st.markdown("## 📊 Dashboard Visualization")

        # --- Match-rate Distribution ---
        fig_dist = px.histogram(
            df,
            x="rating",
            nbins=5,
            title="Match Rate Distribution",
            color_discrete_sequence=["#4F46E5"],
        )
        fig_dist.update_layout(bargap=0.2)
        st.plotly_chart(fig_dist, use_container_width=True)

        # --- Top Strengths & Gaps ---
        st.markdown("### 💪 Top Strengths & 🚧 Gaps Across TGVs")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            mean_scores = df[numeric_cols].mean().sort_values(ascending=False)
            fig_strength = px.bar(
                mean_scores.head(10),
                title="Top Strengths",
                orientation="h",
                color=mean_scores.head(10),
                color_continuous_scale="Blues",
            )
            fig_gap = px.bar(
                mean_scores.tail(10),
                title="Top Gaps",
                orientation="h",
                color=mean_scores.tail(10),
                color_continuous_scale="Reds",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(fig_strength, use_container_width=True)
            with col_b:
                st.plotly_chart(fig_gap, use_container_width=True)

        # --- Benchmark vs Candidate Radar ---
        st.markdown("### 🕸 Benchmark vs Candidate Comparison")
        if len(numeric_cols) >= 3:
            avg_scores = df[numeric_cols].mean().to_dict()
            candidate = top_candidates.iloc[0][numeric_cols].to_dict()
            categories = list(avg_scores.keys())
            benchmark_vals = list(avg_scores.values())
            candidate_vals = list(candidate.values())

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=benchmark_vals, theta=categories, fill="toself", name="Benchmark"))
            fig_radar.add_trace(go.Scatterpolar(r=candidate_vals, theta=categories, fill="toself", name="Candidate"))
            fig_radar.update_layout(title="Benchmark vs Candidate Radar", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- Heatmap Comparison ---
        st.markdown("### 🌡 Heatmap Benchmark vs Candidates")
        if len(numeric_cols) > 0:
            fig_heat = px.imshow(
                df[numeric_cols].corr(),
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ============================================================
        # ACTION BUTTONS
        # ============================================================
        col1, col2 = st.columns(2)

        # --- Generate Job Profile ---
        with col1:
            if st.button("📝 Generate Job Profile"):
                try:
                    prompt = f"""
                    You are an HR expert. Generate a detailed job profile for the role below:

                    *Role:* {role_name}
                    *Level:* {job_level}
                    *Purpose:* {role_purpose}

                    Include:
                    1. Summary
                    2. Key Responsibilities
                    3. Required Skills
                    4. Ideal Background
                    5. KPIs / Performance Metrics
                    """
                    with st.spinner("Generating job profile via OpenRouter..."):
                        job_profile = generate_llm_response(prompt)
                    st.success("✅ Job profile generated successfully!")
                    st.markdown(job_profile)
                except Exception as e:
                    st.error(f"Error: {e}")

        # --- Generate Job Profile + AI Recommendation ---
        with col2:
            if st.button("🚀 Generate Job Profile + AI Recommendation"):
                try:
                    prompt = f"""
                    You are an HR expert. Generate a detailed job profile for:

                    Role: {role_name}
                    Level: {job_level}
                    Purpose: {role_purpose}
                    Include Summary, Responsibilities, Skills, Background, and KPIs.
                    """
                    with st.spinner("Generating job profile..."):
                        job_profile = generate_llm_response(prompt)
                    st.success("✅ Job profile generated!")

                    # Limit candidates
                    top_candidates = top_candidates.head(max_candidates)
                    name_col = next(
                        (c for c in ["fullname", "name", "nama", "employee_name"] if c in top_candidates.columns),
                        top_candidates.columns[0],
                    )
                    kandidat_lines = [
                        f"- {row.get(name_col, 'UNKNOWN')} | Rating: {row.get('rating', 'N/A')}"
                        for _, row in top_candidates.iterrows()
                    ]
                    kandidat_list_text = "\n".join(kandidat_lines)

                    # AI Recommendation
                    prompt2 = f"""
                    Job Profile:
                    {job_profile}

                    Candidates:
                    {kandidat_list_text}

                    Task:
                    Recommend the best-fit candidate for {role_name}, explain why,
                    and provide a short summary of their strengths & development needs.
                    """
                    with st.spinner("Analyzing candidates via OpenRouter..."):
                        result2 = generate_llm_response(prompt2)

                    st.markdown("### 🧠 AI Recommendation & Insights")
                    st.markdown(result2)

                except Exception as e:
                    st.error(f"Error saat rekomendasi: {e}")
else:
    st.info("⬆ Silakan upload file CSV atau Excel berisi data kandidat terlebih dahulu.")

st.markdown("---")
st.caption("Dashboard includes: Match-rate, Strengths/Gaps, Radar, Heatmap, and AI-driven recommendations.")
