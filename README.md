# 🎯 Talent Match Intelligence System  
_A Data-Driven HR Decision Dashboard (Case Study Project)_

---

## 🧩 Overview
This project implements an intelligent **Talent Match Dashboard** designed to analyze employee performance, competencies, and behavioral data to identify top-performing talents.  
It follows a structured data pipeline based on the **Talent Intelligence Case Study** provided in the assignment brief.

The system combines **data analysis**, **SQL logic simulation**, **AI-based interpretation**, and **interactive visualization** — all in a single Streamlit dashboard.

---

## 📊 Project Objectives
| Step | Description |
|------|--------------|
| **Step 1** | Clean and combine multiple Excel sheets (e.g., `performance_yearly`, `competencies_yearly`, etc.) into a single unified dataset `combined_employee_data.csv`. |
| **Step 2** | Analyze the data using SQL logic (implemented through Pandas or PostgreSQL/Supabase). |
| **Step 3** | Create a **Streamlit Dashboard** that visualizes results, ranks employees, and generates AI-based insights. |

---

## ⚙️ Features

✅ Upload and analyze local CSV data (no cloud dependency)  
✅ SQL logic implemented via Pandas (mirroring `employee_analysis` table)  
✅ Weighted success formula combining IQ, competency, leadership, and experience  
✅ Ranked talent list with match rates and TGV (Talent Growth Variables)  
✅ Visual analytics:  
- Match-rate distribution  
- Strengths vs gaps bar plots  
- Benchmark vs Candidate radar  
- Correlation heatmap  
✅ AI-powered insights using **OpenRouter GPT-4o-mini**  
✅ Downloadable CSV output of ranked employees  

---

## 🧱 Project Structure

project_root/
│
├── app/
│ ├── app_dashboard.py # Streamlit main app (dashboard)
│
├── data/
│ ├── combined_employee_data.csv # Cleaned dataset (Step 1 result)
│
├── sql/
│ ├── create_tables.sql # (Optional) Supabase table definition
│ ├── test_queries.sql # Example SQL logic used for analysis
│
├── utils/
│ ├── db_connector.py # (Optional) Supabase connector
│ ├── visualizations.py # Visualization helper functions
│
├── requirements.txt
└── README.md

