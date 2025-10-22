import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_match_distribution(df):
    """Histogram of match rates"""
    fig = px.histogram(df, x='final_match_rate', 
                       title='Distribution of Candidate Match Rates',
                       labels={'final_match_rate': 'Match Rate (%)'},
                       nbins=30)
    return fig

def create_tgv_radar(df, employee_id):
    """Radar chart for individual candidate TGV scores"""
    candidate = df[df['employee_id'] == employee_id].iloc[0]
    
    categories = ['Cognitive', 'Leadership', 'Execution', 'Work Pref', 'Interpersonal']
    values = [
        candidate['cognitive_match'],
        candidate['leadership_match'],
        candidate['execution_match'],
        candidate.get('work_pref_match', 75),
        candidate.get('interpersonal_match', 80)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=candidate['fullname']
    ))
    fig.update_layout(title=f"TGV Profile: {candidate['fullname']}")
    return fig

def create_top_candidates_bar(df, top_n=10):
    """Bar chart of top candidates"""
    top_df = df.nlargest(top_n, 'final_match_rate')
    fig = px.bar(top_df, x='fullname', y='final_match_rate',
                 title=f'Top {top_n} Matching Candidates',
                 labels={'final_match_rate': 'Match Rate (%)', 'fullname': 'Candidate'})
    fig.update_xaxis(tickangle=-45)
    return fig