# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

st.set_page_config(page_title="IPL Analytics & ML Dashboard", page_icon="🏏", layout="wide")

# ==========================================
# 1. DATA LOADING (Cached)
# ==========================================
@st.cache_data
def load_data():
    needed_cols = ['match_id', 'venue', 'batting_team', 'bowling_team', 'innings', 'runs_total', 'bowler_wicket', 'over', 'balls_faced', 'runs_batter', 'batter', 'bowler', 'valid_ball', 'toss_winner', 'toss_decision', 'match_won_by', 'stage', 'wicket_kind', 'season', 'runs_bowler']
    df = pd.read_csv("IPL.zip", low_memory=False, usecols=lambda x: x in needed_cols)
    df.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru', inplace=True)
    
    venue_mapping = {
        'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
        'Feroz Shah Kotla': 'Arun Jaitley Stadium',
        'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
        'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Eden Gardens, Kolkata': 'Eden Gardens',
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
        'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
        'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
        'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
        'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium',
        'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association Stadium, Mohali',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
        'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium, Ahmedabad',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
        'Zayed Cricket Stadium, Abu Dhabi': 'Sheikh Zayed Stadium'
    }
    df['venue'] = df['venue'].replace(venue_mapping)
    
    if 'year' in df.columns:
        df = df[df['year'].astype(str) != '2026']
    elif 'season' in df.columns:
        df = df[df['season'].astype(str) != '2026']
        
    matches = df.drop_duplicates(subset='match_id').copy()
    
    stats = {}
    stats['team_wins'] = matches['match_won_by'].value_counts().head(10).reset_index()
    stats['team_wins'].columns = ['Team', 'Wins']
    
    col = 'season' if 'season' in df.columns else 'year'
    
    finals = matches[matches['stage'] == 'Final'].copy().sort_values(col, ascending=False)
    stats['championships'] = finals[[col, 'match_won_by']].rename(columns={col: 'Season', 'match_won_by': 'Champion'})
    
    m2 = matches.dropna(subset=['toss_winner', 'match_won_by']).copy()
    m2['tw'] = m2['toss_winner'] == m2['match_won_by']
    toss_c = m2['tw'].value_counts()
    stats['toss_effect'] = pd.DataFrame({'Outcome': ['Toss Winner Won', 'Toss Winner Lost'], 'Count': [int(toss_c.get(True, 0)), int(toss_c.get(False, 0))]})
    
    td = matches['toss_decision'].value_counts().reset_index()
    td.columns = ['Decision', 'Count']
    stats['toss_decision'] = td
    
    s_bat = df.groupby('batter').agg(total_runs=('runs_batter','sum'), total_balls=('balls_faced','sum')).reset_index()
    s_bat['strike_rate'] = (s_bat['total_runs'] / s_bat['total_balls'] * 100).round(2)
    stats['top_batters'] = s_bat[s_bat['total_balls'] >= 500].sort_values('strike_rate', ascending=False).head(10)
    
    v_bowl = df[df['valid_ball'] == 1]
    s_bowl = v_bowl.groupby('bowler').agg(total_runs=('runs_bowler','sum'), total_balls=('valid_ball','sum'), total_wickets=('bowler_wicket','sum')).reset_index()
    s_bowl['economy'] = (s_bowl['total_runs'] / s_bowl['total_balls'] * 6).round(2)
    stats['top_bowlers'] = s_bowl[s_bowl['total_balls'] >= 1000].sort_values('economy').head(10)

    pp_bat = df[df['over'] <= 6].groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(8).reset_index(name='runs')
    dt_bat = df[df['over'] >= 17].groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(8).reset_index(name='runs')
    stats['pp_bat'] = pp_bat
    stats['dt_bat'] = dt_bat
    
    d_types = df[df['wicket_kind'].notna()]['wicket_kind'].value_counts().reset_index()
    d_types.columns = ['Type', 'Count']
    stats['dismissals'] = d_types
    
    ts = df.groupby(['season','match_id','batting_team'])['runs_total'].sum().reset_index()
    avg_scores = ts.groupby('season')['runs_total'].mean().round(1).reset_index(name='Avg Score')
    stats['season_scores'] = avg_scores
    
    fours = df[df['runs_batter'] == 4].groupby('batter').size().reset_index(name='fours').sort_values('fours', ascending=False).head(10)
    sixes = df[df['runs_batter'] == 6].groupby('batter').size().reset_index(name='sixes').sort_values('sixes', ascending=False).head(10)
    stats['fours'] = fours
    stats['sixes'] = sixes
    
    orange_caps = df.groupby([col, 'batter'])['runs_batter'].sum().reset_index()
    idx_orange = orange_caps.groupby(col)['runs_batter'].idxmax()
    orange_res = orange_caps.loc[idx_orange].sort_values(col, ascending=False)
    
    purple_caps = df.groupby([col, 'bowler'])['bowler_wicket'].sum().reset_index()
    idx_purple = purple_caps.groupby(col)['bowler_wicket'].idxmax()
    purple_res = purple_caps.loc[idx_purple].sort_values(col, ascending=False)
    
    stats['orange_cap'] = orange_res
    stats['purple_cap'] = purple_res
    
    finals_dict = dict(zip(finals[col].astype(str), finals['match_won_by']))
    orange_dict = dict(zip(orange_res[col].astype(str), orange_res['batter']))
    purple_dict = dict(zip(purple_res[col].astype(str), purple_res['bowler']))
    all_seasons = sorted(list(set(finals_dict.keys()) | set(orange_dict.keys()) | set(purple_dict.keys())), reverse=True)
    
    hist = []
    for s in all_seasons:
        hist.append({
            'Season': s,
            'Champion': finals_dict.get(s, 'N/A'),
            'Orange Cap': orange_dict.get(s, 'N/A'),
            'Purple Cap': purple_dict.get(s, 'N/A')
        })
    stats['history'] = pd.DataFrame(hist)
    
    return df, matches, stats

df, matches, stats = load_data()

# ==========================================
# 2. MODEL LOADING (Cached)
# ==========================================
@st.cache_resource
def load_models():
    score_model_data, win_model_data = None, None
    if os.path.exists('score_model_full.joblib'):
        score_model_data = joblib.load('score_model_full.joblib')
    if os.path.exists('win_model_full.joblib'):
        win_model_data = joblib.load('win_model_full.joblib')
        
    cluster_data = {'batters': None, 'bowlers': None, 'allrounders': None}
    
    try:
        entry_overs = df.groupby(['match_id', 'batter'])['over'].min().groupby('batter').mean().reset_index().rename(columns={'over': 'avg_entry_over'})
        bs = df.groupby('batter').agg(total_runs=('runs_batter', 'sum'), total_balls=('balls_faced', 'sum'), matches=('match_id', 'nunique')).reset_index()
        bs = bs.merge(entry_overs, on='batter', how='left')
        bs = bs[bs['total_balls'] >= 200].copy()
        bs['strike_rate'], bs['avg_runs'] = (bs['total_runs'] / bs['total_balls'] * 100).round(2), (bs['total_runs'] / bs['matches']).round(2)
        scaler_b = StandardScaler()
        X_b = scaler_b.fit_transform(bs[['avg_entry_over', 'strike_rate', 'avg_runs']])
        kmeans_b = KMeans(n_clusters=3, random_state=42, n_init=10)
        bs['cluster'] = kmeans_b.fit_predict(X_b)
        cluster_entry = bs.groupby('cluster')['avg_entry_over'].mean().sort_values()
        roles_b = ['🏏 Opener', '🛡️ Middle Order', '⚡ Finisher']
        labels_map_b = {c: roles_b[i] for i, (c, _) in enumerate(cluster_entry.items())}
        bs['role'] = bs['cluster'].map(labels_map_b)
        cluster_data['batters'] = bs

        v_balls = df[df['valid_ball'] == 1]
        mid_counts = df[(df['over'] > 6) & (df['over'] < 16)].groupby('bowler').size().reset_index(name='is_mid')
        bowls = v_balls.groupby('bowler').agg(total_runs=('runs_bowler', 'sum'), total_balls=('valid_ball', 'sum'), total_wickets=('bowler_wicket', 'sum')).reset_index()
        bowls = bowls.merge(mid_counts, on='bowler', how='left').fillna(0)
        bowls = bowls[bowls['total_balls'] >= 300].copy()
        bowls['economy'] = (bowls['total_runs'] / bowls['total_balls'] * 6).round(2)
        bowls['strike_rate'] = (bowls['total_balls'] / bowls['total_wickets'].replace(0, np.nan)).round(2)
        bowls['strike_rate'].fillna(bowls['strike_rate'].max(), inplace=True)
        bowls['mid_pct'] = bowls['is_mid'] / bowls['total_balls']
        scaler_bw = StandardScaler()
        X_bw = scaler_bw.fit_transform(bowls[['economy', 'strike_rate', 'mid_pct']])
        kmeans_bw = KMeans(n_clusters=4, random_state=42, n_init=10)
        bowls['cluster'] = kmeans_bw.fit_predict(X_bw)
        cluster_stats = bowls.groupby('cluster').agg(avg_mid_pct=('mid_pct', 'mean'), avg_econ=('economy', 'mean'), avg_sr=('strike_rate', 'mean')).reset_index().sort_values('avg_mid_pct', ascending=False)
        spinners, pacers = cluster_stats.head(2).sort_values('avg_econ'), cluster_stats.tail(2).sort_values('avg_sr')
        labels_map_bw = {spinners.iloc[0]['cluster']: '🌀 Off Spinner', spinners.iloc[1]['cluster']: '🪄 Leg Spinner', pacers.iloc[0]['cluster']: '🚀 Pacer', pacers.iloc[1]['cluster']: '🎯 Medium Pacer'}
        bowls['role'] = bowls['cluster'].map(labels_map_bw)
        cluster_data['bowlers'] = bowls

        ar = pd.merge(bs[['batter', 'total_runs', 'strike_rate', 'avg_runs']], bowls[['bowler', 'total_wickets', 'economy', 'strike_rate']], left_on='batter', right_on='bowler', how='inner')
        ar.rename(columns={'batter': 'player', 'strike_rate_x': 'bat_sr', 'strike_rate_y': 'bowl_sr'}, inplace=True)
        scaler_ar = StandardScaler()
        X_ar = scaler_ar.fit_transform(ar[['bat_sr', 'avg_runs', 'economy', 'total_wickets']])
        kmeans_ar = KMeans(n_clusters=3, random_state=42, n_init=10)
        ar['cluster'] = kmeans_ar.fit_predict(X_ar)

        cluster_bat_avg = ar.groupby('cluster')['avg_runs'].mean().sort_values()
        roles_ar = ['🎳 Bowling Allrounder', '🌱 Utility Player', '🏏 Batting Allrounder']
        labels_map_ar = {c: roles_ar[i] for i, (c, _) in enumerate(cluster_bat_avg.items())}
        ar['role'] = ar['cluster'].map(labels_map_ar)
        cluster_data['allrounders'] = ar
    except Exception as e:
        st.error(f"Clustering error: {e}")
        
    return score_model_data, win_model_data, cluster_data

score_mdl, win_mdl, clusters = load_models()

# ==========================================
# 3. UI LAYOUT
# ==========================================
st.title("🏏 IPL Analytics & ML Dashboard")
st.markdown("Interactive insights powered by Streamlit, scikit-learn, and XGBoost.")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Matches", f"{len(matches):,}")
with col2:
    st.metric("Seasons", f"{matches['season'].nunique() if 'season' in matches.columns else matches['year'].nunique()}")
with col3:
    st.metric("Teams", f"{len(stats['team_wins'])}")
with col4:
    st.metric("Most Wins", f"{stats['team_wins'].iloc[0]['Team']}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🏆 Teams", "🪙 Toss", "🏏 Batting", "🎳 Bowling", "📅 Seasons", "📚 History", 
    "🤖 Score Predictor", "🎯 Win Predictor", "🔬 Player Clusters"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most Wins by Team")
        fig = px.bar(stats['team_wins'], x='Team', y='Wins', color_discrete_sequence=['#f59e0b'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Toss Decision Preference")
        fig = px.bar(stats['toss_decision'], x='Decision', y='Count', color_discrete_sequence=['#14b8a6'])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Does Winning the Toss Help?")
        fig = px.pie(stats['toss_effect'], names='Outcome', values='Count', color_discrete_sequence=['#14b8a6', '#f43f5e'], hole=0.6)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Bat vs Field After Toss")
        st.plotly_chart(px.bar(stats['toss_decision'], x='Decision', y='Count', color_discrete_sequence=['#14b8a6']), use_container_width=True, key="toss_decision_tab2")

with tab3:
    st.subheader("Top 10 Batters by Strike Rate (min 500 balls)")
    fig = px.bar(stats['top_batters'], x='batter', y='strike_rate', hover_data=['total_runs'], color_discrete_sequence=['#8b5cf6'])
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Highest 4 Hitters")
        st.plotly_chart(px.bar(stats['fours'], x='batter', y='fours', color_discrete_sequence=['#14b8a6']), use_container_width=True)
    with c2:
        st.subheader("Highest 6 Hitters")
        st.plotly_chart(px.bar(stats['sixes'], x='batter', y='sixes', color_discrete_sequence=['#f59e0b']), use_container_width=True)
        
    st.subheader("Orange Cap Winners")
    st.dataframe(stats['orange_cap'], hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Top 10 Bowlers by Economy (min 1000 balls)")
    fig = px.bar(stats['top_bowlers'], x='bowler', y='economy', hover_data=['total_wickets'], color_discrete_sequence=['#10b981'])
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dismissal Types")
        st.plotly_chart(px.bar(stats['dismissals'], x='Type', y='Count', color_discrete_sequence=['#f43f5e']), use_container_width=True)
    with c2:
        st.subheader("Purple Cap Winners")
        st.dataframe(stats['purple_cap'], hide_index=True, use_container_width=True)

with tab5:
    st.subheader("Season-wise Average Score")
    fig = px.line(stats['season_scores'], x='season', y='Avg Score', markers=True, color_discrete_sequence=['#6366f1'])
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Tournament History")
    st.dataframe(stats['history'], hide_index=True, use_container_width=True)

with tab7:
    st.header("Score Predictor (XGBoost)")
    if score_mdl:
        sm = score_mdl['model']
        le_t = score_mdl['le_team']
        le_v = score_mdl['le_venue']
        r2 = score_mdl['r2']
        
        st.success(f"Model loaded successfully. R² Score: {r2}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            bat_team = st.selectbox("Batting Team", le_t.classes_)
        with c2:
            bowl_team = st.selectbox("Bowling Team", le_t.classes_, index=1)
        with c3:
            venue = st.selectbox("Venue", le_v.classes_)
            
        c4, c5, c6 = st.columns(3)
        with c4:
            current_over = st.number_input("Current Over", min_value=1.0, max_value=20.0, value=10.0, step=0.1)
        with c5:
            current_runs = st.number_input("Current Runs", min_value=0, max_value=400, value=85)
        with c6:
            current_wickets = st.number_input("Current Wickets", min_value=0, max_value=10, value=2)
            
        if st.button("🎯 Predict Final Score"):
            v_enc = le_v.transform([venue])[0]
            b_enc = le_t.transform([bat_team])[0]
            bw_enc = le_t.transform([bowl_team])[0]
            crr = current_runs / max(1, current_over)
            
            if current_over >= 20:
                pred = int(current_runs)
            else:
                pred = int(max(current_runs, min(350, round(sm.predict([[current_over, current_runs, current_wickets, crr, v_enc, b_enc, bw_enc]])[0]))))
                
            st.metric("Predicted Final Score", pred)
    else:
        st.warning("Score model not found. Please train it first or ensure joblib files exist.")

with tab8:
    st.header("Win Probability Predictor (XGBoost)")
    if win_mdl:
        wm = win_mdl['model']
        le_t_w = win_mdl['le_team']
        le_v_w = win_mdl['le_venue']
        le_d_w = win_mdl['le_dec']
        h2h_d = win_mdl['h2h']
        h2h_v_d = win_mdl['h2h_v']
        venue_c_wr = win_mdl['venue_chase_wr']
        curr_form = win_mdl['current_form']
        acc = win_mdl['acc']
        
        st.success(f"Model loaded successfully. Test Accuracy: {acc}%")
        
        c1, c2 = st.columns(2)
        with c1:
            t1 = st.selectbox("Team 1 (Batting)", win_mdl['teams'])
            toss_w = st.selectbox("Toss Winner", win_mdl['teams'])
        with c2:
            t2 = st.selectbox("Team 2 (Bowling)", win_mdl['teams'], index=1)
            toss_d = st.selectbox("Toss Decision", ['bat', 'field'])
            
        v_win = st.selectbox("Match Venue", win_mdl['venues'])
        
        if st.button("🏏 Predict Winner"):
            t1_e = int(le_t_w.transform([t1])[0])
            t2_e = int(le_t_w.transform([t2])[0])
            v_e = int(le_v_w.transform([v_win])[0]) if v_win in le_v_w.classes_ else 0
            d_e = int(le_d_w.transform([toss_d])[0])
            
            h2h = h2h_d.get((t1, t2), 0.5)
            h2h_v = h2h_v_d.get((t1, t2, v_win), 0.5)
            v_wr = venue_c_wr.get(v_win, 0.5)
            t_imp = v_wr if toss_d == 'field' else (1.0 - v_wr)
            t1_cf = curr_form.get(t1, 0.5)
            t2_cf = curr_form.get(t2, 0.5)
            
            proba = wm.predict_proba([[t1_e, t2_e, v_e, d_e, h2h, h2h_v, t_imp, t1_cf, t2_cf]])[0]
            
            st.write("### Win Probability")
            c3, c4 = st.columns(2)
            c3.metric(t1, f"{round(proba[1]*100, 1)}%")
            c4.metric(t2, f"{round(proba[0]*100, 1)}%")
            
            st.info(f"**Predicted Winner:** {t1 if proba[1] > 0.5 else t2}")
    else:
        st.warning("Win model not found.")

with tab9:
    st.header("Player Role Clusters (K-Means)")
    if clusters['batters'] is not None:
        cl_tab1, cl_tab2, cl_tab3 = st.tabs(["Batters", "Bowlers", "Allrounders"])
        
        with cl_tab1:
            st.dataframe(clusters['batters'][['batter', 'role', 'strike_rate', 'avg_runs', 'total_runs']], hide_index=True, use_container_width=True)
            st.bar_chart(clusters['batters']['role'].value_counts())
            
        with cl_tab2:
            st.dataframe(clusters['bowlers'][['bowler', 'role', 'economy', 'strike_rate', 'total_wickets']], hide_index=True, use_container_width=True)
            st.bar_chart(clusters['bowlers']['role'].value_counts())
            
        with cl_tab3:
            st.dataframe(clusters['allrounders'][['player', 'role', 'bat_sr', 'avg_runs', 'economy', 'total_wickets']], hide_index=True, use_container_width=True)
            st.bar_chart(clusters['allrounders']['role'].value_counts())
    else:
        st.warning("Cluster data not available.")
