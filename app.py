from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import gc

# Global storage for aggregated chart data
stats_data = {}

app = Flask(__name__)

# ══════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════
print("Loading IPL data...")
# Load only necessary columns to save RAM
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

print(f"OK: {len(df):,} rows | {len(matches):,} matches")

# ══════════════════════════════════════════════════════
#  ML MODEL 1 — Score Predictor (XGBoost)
# ══════════════════════════════════════════════════════
print("Loading Score Predictor...")
try:
    if os.path.exists('score_model_full.joblib'):
        data = joblib.load('score_model_full.joblib')
        score_model = data['model']
        score_le_team = data['le_team']
        score_le_venue = data['le_venue']
        score_r2 = data['r2']
        print(f"   Loaded from disk (R² = {score_r2})")
    else:
        print("   Training from scratch...")
        match_info_sc = df[['match_id', 'venue', 'batting_team', 'bowling_team', 'innings']].drop_duplicates()
        over_stats = df.groupby(['match_id', 'innings', 'over']).agg({
            'runs_total': 'sum', 'bowler_wicket': 'sum'
        }).reset_index()
        over_stats = over_stats.sort_values(['match_id', 'innings', 'over'])
        over_stats['current_runs'] = over_stats.groupby(['match_id', 'innings'])['runs_total'].cumsum()
        over_stats['current_wickets'] = over_stats.groupby(['match_id', 'innings'])['bowler_wicket'].cumsum()
        final_scores = df.groupby(['match_id', 'innings'])['runs_total'].sum().reset_index().rename(columns={'runs_total': 'final_score'})
        model_data_sc = over_stats.merge(final_scores, on=['match_id', 'innings'])
        model_data_sc = model_data_sc.merge(match_info_sc, on=['match_id', 'innings'])
        model_data_sc['completed_overs'] = model_data_sc['over'] + 1
        model_data_sc['crr'] = model_data_sc['current_runs'] / model_data_sc['completed_overs']
        model_data_sc = model_data_sc[model_data_sc['innings'] == 1]
        score_le_venue, score_le_team = LabelEncoder(), LabelEncoder()
        all_teams_score = pd.concat([model_data_sc['batting_team'], model_data_sc['bowling_team']]).unique()
        score_le_team.fit(all_teams_score)
        score_le_venue.fit(model_data_sc['venue'])
        model_data_sc['venue_enc'] = score_le_venue.transform(model_data_sc['venue'])
        model_data_sc['bat_enc'] = score_le_team.transform(model_data_sc['batting_team'])
        model_data_sc['bowl_enc'] = score_le_team.transform(model_data_sc['bowling_team'])
        X_sc = model_data_sc[['completed_overs', 'current_runs', 'current_wickets', 'crr', 'venue_enc', 'bat_enc', 'bowl_enc']]
        y_sc = model_data_sc['final_score']
        score_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        score_model.fit(X_sc, y_sc)
        score_r2 = round(score_model.score(X_sc, y_sc), 3)
        joblib.dump({'model': score_model, 'le_team': score_le_team, 'le_venue': score_le_venue, 'r2': score_r2}, 'score_model_full.joblib')
        print(f"   Done (R² = {score_r2})")
except Exception as e:
    print(f"   Error: {e}")

# ══════════════════════════════════════════════════════
#  ML MODEL 2 — Win Predictor (Advanced XGBoost)
# ══════════════════════════════════════════════════════
print("Loading Win Predictor...")
win_model = None
win_acc = 0
try:
    if os.path.exists('win_model_full.joblib'):
        data = joblib.load('win_model_full.joblib')
        win_model, le_team, le_venue, le_dec = data['model'], data['le_team'], data['le_venue'], data['le_dec']
        win_acc, h2h_dict, h2h_v_dict, venue_chase_wr = data['acc'], data['h2h'], data['h2h_v'], data['venue_chase_wr']
        current_form_dict, teams_list, venues_list = data['current_form'], data['teams'], data['venues']
        print(f"   Loaded from disk (Accuracy = {win_acc}%)")
    else:
        print("   Training from scratch...")
        m = matches.dropna(subset=['toss_winner', 'toss_decision', 'venue', 'batting_team', 'bowling_team', 'match_won_by']).copy()
        all_teams = pd.concat([m['batting_team'], m['bowling_team'], m['toss_winner']]).unique()
        le_team, le_venue, le_dec = LabelEncoder(), LabelEncoder(), LabelEncoder()
        le_team.fit(all_teams)
        le_venue.fit(m['venue'])
        le_dec.fit(m['toss_decision'])
        venue_chase_wr = {}
        for v in m['venue'].unique():
            v_matches = m[m['venue'] == v]
            bat_first_won = len(v_matches[(v_matches['toss_decision'] == 'bat') & (v_matches['toss_winner'] == v_matches['match_won_by'])]) + \
                            len(v_matches[(v_matches['toss_decision'] == 'field') & (v_matches['toss_winner'] != v_matches['match_won_by'])])
            venue_chase_wr[v] = (len(v_matches) - bat_first_won) / len(v_matches) if len(v_matches) > 0 else 0.5
        m['v_chase_wr'] = m['venue'].map(lambda x: venue_chase_wr.get(x, 0.5))
        m['toss_impact'] = m.apply(lambda row: row['v_chase_wr'] if row['toss_decision'] == 'field' else 1.0 - row['v_chase_wr'], axis=1)
        current_form_dict, team_history = {}, {team: [] for team in all_teams}
        t1_cf, t2_cf = [], []
        m_sorted = m.sort_values('date') if 'date' in m.columns else m.copy()
        for _, row in m_sorted.iterrows():
            t1, t2 = row['batting_team'], row['bowling_team']
            h1, h2 = team_history[t1][-5:], team_history[t2][-5:]
            t1_cf.append(sum(h1)/len(h1) if h1 else 0.5)
            t2_cf.append(sum(h2)/len(h2) if h2 else 0.5)
            team_history[t1].append(1 if row['match_won_by'] == t1 else 0)
            team_history[t2].append(1 if row['match_won_by'] == t2 else 0)
        m_sorted['t1_cf'], m_sorted['t2_cf'] = t1_cf, t2_cf
        m = m_sorted.copy()
        for team, history in team_history.items(): current_form_dict[team] = sum(history[-5:])/len(history[-5:]) if history[-5:] else 0.5
        h2h_dict, h2h_v_dict = {}, {}
        for t1 in all_teams:
            for t2 in all_teams:
                if t1 == t2: continue
                sub = m[((m['batting_team']==t1)&(m['bowling_team']==t2)) | ((m['batting_team']==t2)&(m['bowling_team']==t1))]
                if len(sub) > 0:
                    h2h_dict[(t1, t2)] = len(sub[sub['match_won_by']==t1]) / len(sub)
                    for v in m['venue'].unique():
                        v_sub = sub[sub['venue'] == v]
                        h2h_v_dict[(t1, t2, v)] = len(v_sub[v_sub['match_won_by']==t1]) / len(v_sub) if len(v_sub) > 0 else 0.5
                else: h2h_dict[(t1, t2)] = 0.5
        m['h2h'] = m.apply(lambda r: h2h_dict.get((r['batting_team'], r['bowling_team']), 0.5), axis=1)
        m['h2h_v'] = m.apply(lambda r: h2h_v_dict.get((r['batting_team'], r['bowling_team'], r['venue']), 0.5), axis=1)
        m['t1_enc'], m['t2_enc'], m['v_enc'], m['d_enc'] = le_team.transform(m['batting_team']), le_team.transform(m['bowling_team']), le_venue.transform(m['venue']), le_dec.transform(m['toss_decision'])
        m['target'] = (m['match_won_by'] == m['batting_team']).astype(int)
        feats = ['t1_enc', 't2_enc', 'v_enc', 'd_enc', 'h2h', 'h2h_v', 'toss_impact', 't1_cf', 't2_cf']
        Xtr, Xte, ytr, yte = train_test_split(m[feats], m['target'], test_size=0.3, random_state=42)
        win_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        win_model.fit(Xtr, ytr)
        win_acc = round(accuracy_score(yte, win_model.predict(Xte)) * 100, 1)
        teams_list, venues_list = sorted(all_teams.tolist()), sorted(m['venue'].unique().tolist())
        joblib.dump({'model': win_model, 'le_team': le_team, 'le_venue': le_venue, 'le_dec': le_dec, 'acc': win_acc, 'h2h': h2h_dict, 'h2h_v': h2h_v_dict, 'venue_chase_wr': venue_chase_wr, 'current_form': current_form_dict, 'teams': teams_list, 'venues': venues_list}, 'win_model_full.joblib')
        print(f"   Done (Accuracy = {win_acc}%)")
except Exception as e:
    print(f"   Error: {e}")

# ══════════════════════════════════════════════════════
#  ML MODEL 3 — Player Clustering (K-Means)
# ══════════════════════════════════════════════════════
print("Training Player Clusters (K-Means)...")
cluster_data = {'batters': None, 'bowlers': None, 'allrounders': None}
try:
    # --- BATTERS ---
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

    # --- BOWLERS ---
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

    # --- ALLROUNDERS ---
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

    # --- PRE-AGGREGATE CHART DATA ---
    print("   Pre-aggregating chart data...")
    stats_data['team_wins'] = matches['match_won_by'].value_counts().head(10).to_dict()
    
    col = 'season' if 'season' in df.columns else 'year'
    
    # Championships
    finals = matches[matches['stage'] == 'Final'].copy()
    finals = finals.sort_values(col, ascending=False)
    stats_data['championships'] = {
        'seasons': finals[col].astype(str).tolist(),
        'teams': finals['match_won_by'].tolist()
    }
    
    # Toss Effect
    m2 = matches.dropna(subset=['toss_winner', 'match_won_by']).copy()
    m2['tw'] = m2['toss_winner'] == m2['match_won_by']
    toss_c = m2['tw'].value_counts()
    stats_data['toss_effect'] = {'labels': ['Toss Winner Won', 'Toss Winner Lost'], 'values': [int(toss_c.get(True, 0)), int(toss_c.get(False, 0))]}
    
    stats_data['toss_decision'] = matches['toss_decision'].value_counts().to_dict()
    
    # Top Batters
    s_bat = df.groupby('batter').agg(total_runs=('runs_batter','sum'), total_balls=('balls_faced','sum')).reset_index()
    s_bat['strike_rate'] = (s_bat['total_runs'] / s_bat['total_balls'] * 100).round(2)
    top_bat = s_bat[s_bat['total_balls'] >= 500].sort_values('strike_rate', ascending=False).head(10)
    stats_data['top_batters'] = {
        'batters': top_bat['batter'].tolist(),
        'strike_rates': top_bat['strike_rate'].tolist(),
        'total_runs': top_bat['total_runs'].tolist()
    }
    
    # Top Bowlers
    v_bowl = df[df['valid_ball'] == 1]
    s_bowl = v_bowl.groupby('bowler').agg(total_runs=('runs_bowler','sum'), total_balls=('valid_ball','sum'), total_wickets=('bowler_wicket','sum')).reset_index()
    s_bowl['economy'] = (s_bowl['total_runs'] / s_bowl['total_balls'] * 6).round(2)
    top_bowl = s_bowl[s_bowl['total_balls'] >= 1000].sort_values('economy').head(10)
    stats_data['top_bowlers'] = {
        'bowlers': top_bowl['bowler'].tolist(),
        'economy': top_bowl['economy'].tolist(),
        'wickets': top_bowl['total_wickets'].tolist()
    }

    # Additional Stats from provided code
    pp_bat = df[df['over'] <= 6].groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(8)
    dt_bat = df[df['over'] >= 17].groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(8)
    stats_data['phase_runs'] = {
        'pp_batters': pp_bat.index.tolist(), 'pp_runs': pp_bat.values.tolist(),
        'dt_batters': dt_bat.index.tolist(), 'dt_runs': dt_bat.values.tolist()
    }

    dismissals_data = df[df['wicket_kind'].notna()]['wicket_kind'].value_counts()
    stats_data['dismissals'] = {'types': dismissals_data.index.tolist(), 'counts': dismissals_data.values.tolist()}

    ts = df.groupby(['season','match_id','batting_team'])['runs_total'].sum().reset_index()
    avg_scores = ts.groupby('season')['runs_total'].mean().round(1)
    stats_data['season_scores'] = {'seasons': [str(s) for s in avg_scores.index.tolist()], 'avg_scores': avg_scores.values.tolist()}

    fours = df[df['runs_batter'] == 4].groupby('batter').size().reset_index(name='fours')
    sixes = df[df['runs_batter'] == 6].groupby('batter').size().reset_index(name='sixes')
    top_4s = fours.sort_values('fours', ascending=False).head(10)
    top_6s = sixes.sort_values('sixes', ascending=False).head(10)
    stats_data['boundaries'] = {
        'top_4s': {'batters': top_4s['batter'].tolist(), 'count': top_4s['fours'].tolist()},
        'top_6s': {'batters': top_6s['batter'].tolist(), 'count': top_6s['sixes'].tolist()}
    }

    pp_bowl = df[df['over'] <= 6].groupby('bowler')['bowler_wicket'].sum().sort_values(ascending=False).head(8)
    dt_bowl = df[df['over'] >= 17].groupby('bowler')['bowler_wicket'].sum().sort_values(ascending=False).head(8)
    stats_data['phase_wickets'] = {
        'pp_bowlers': pp_bowl.index.tolist(), 'pp_wickets': pp_bowl.values.tolist(),
        'dt_bowlers': dt_bowl.index.tolist(), 'dt_wickets': dt_bowl.values.tolist()
    }

    # Orange & Purple Caps
    orange_caps = df.groupby([col, 'batter'])['runs_batter'].sum().reset_index()
    idx_orange = orange_caps.groupby(col)['runs_batter'].idxmax()
    orange_res = orange_caps.loc[idx_orange].sort_values(col)
    stats_data['orange_cap'] = {
        'seasons': orange_res[col].astype(str).tolist(),
        'batters': orange_res['batter'].tolist(),
        'runs': orange_res['runs_batter'].tolist()
    }

    purple_caps = df.groupby([col, 'bowler'])['bowler_wicket'].sum().reset_index()
    idx_purple = purple_caps.groupby(col)['bowler_wicket'].idxmax()
    purple_res = purple_caps.loc[idx_purple].sort_values(col)
    stats_data['purple_cap'] = {
        'seasons': purple_res[col].astype(str).tolist(),
        'bowlers': purple_res['bowler'].tolist(),
        'wickets': purple_res['bowler_wicket'].tolist()
    }

    # History Table
    finals_dict = dict(zip(finals[col].astype(str), finals['match_won_by']))
    orange_dict = dict(zip(orange_res[col].astype(str), orange_res['batter']))
    purple_dict = dict(zip(purple_res[col].astype(str), purple_res['bowler']))
    all_seasons = sorted(list(set(finals_dict.keys()) | set(orange_dict.keys()) | set(purple_dict.keys())), reverse=True)
    history_list = []
    for s in all_seasons:
        history_list.append({
            'season': s,
            'champion': finals_dict.get(s, 'N/A'),
            'orange_cap': orange_dict.get(s, 'N/A'),
            'purple_cap': purple_dict.get(s, 'N/A')
        })
    stats_data['history'] = history_list

    print("   Memory optimization complete.")
except Exception as e:
    print(f"   Error in ML/Stats: {e}")

# Clear large dataframes to free up RAM for the API
try:
    del df
    del matches
    gc.collect()
    print("   Raw Dataframes deleted to save RAM.")
except:
    pass

print("OK: All models ready!\n")

# ══════════════════════════════════════════════════════
#  ANALYSIS ROUTES
# ══════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/team_wins')
def team_wins():
    d = stats_data.get('team_wins', {})
    return jsonify({'teams': list(d.keys()), 'wins': list(d.values())})

@app.route('/api/championships')
def championships():
    return jsonify(stats_data.get('championships', {'seasons': [], 'teams': []}))

@app.route('/api/toss_effect')
def toss_effect():
    return jsonify(stats_data.get('toss_effect', {'labels': [], 'values': []}))

@app.route('/api/toss_decision')
def toss_decision():
    d = stats_data.get('toss_decision', {})
    return jsonify({'decisions': list(d.keys()), 'counts': list(d.values())})

@app.route('/api/top_batters')
def top_batters():
    return jsonify(stats_data.get('top_batters', {'batters': [], 'strike_rates': [], 'total_runs': []}))

@app.route('/api/top_bowlers')
def top_bowlers():
    return jsonify(stats_data.get('top_bowlers', {'bowlers': [], 'economy': [], 'wickets': []}))

@app.route('/api/phase_runs')
def phase_runs():
    return jsonify(stats_data.get('phase_runs', {}))

@app.route('/api/dismissals')
def dismissals():
    return jsonify(stats_data.get('dismissals', {}))

@app.route('/api/season_scores')
def season_scores():
    return jsonify(stats_data.get('season_scores', {}))

@app.route('/api/boundaries')
def boundaries():
    return jsonify(stats_data.get('boundaries', {}))

@app.route('/api/orange_cap')
def orange_cap():
    return jsonify(stats_data.get('orange_cap', {'seasons': [], 'batters': [], 'runs': []}))

@app.route('/api/phase_wickets')
def phase_wickets():
    return jsonify(stats_data.get('phase_wickets', {}))

@app.route('/api/purple_cap')
def purple_cap():
    return jsonify(stats_data.get('purple_cap', {'seasons': [], 'bowlers': [], 'wickets': []}))

@app.route('/api/history')
def history():
    return jsonify(stats_data.get('history', []))

# ══════════════════════════════════════════════════════
#  ML ROUTES
# ══════════════════════════════════════════════════════

@app.route('/api/ml/predict_score', methods=['POST'])
def predict_score():
    try:
        data      = request.json
        c_overs   = float(data['current_over'])
        c_runs    = float(data['current_runs'])
        c_wkts    = float(data['current_wickets'])
        venue     = data['venue']
        bat_team  = data['batting_team']
        bowl_team = data['bowling_team']
        
        v_enc = score_le_venue.transform([venue])[0]
        b_enc = score_le_team.transform([bat_team])[0]
        bw_enc = score_le_team.transform([bowl_team])[0]
        crr = c_runs / max(1, c_overs)
        
        if c_overs >= 20:
            predicted = int(c_runs)
        else:
            predicted = int(max(c_runs, min(350, round(score_model.predict([[c_overs, c_runs, c_wkts, crr, v_enc, b_enc, bw_enc]])[0]))))
            
        return jsonify({
            'predicted_score': predicted,
            'r2_score': score_r2,
            'model': 'XGBoost',
            'insight': f'Projected from over {int(c_overs)}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/ml/teams')
def get_teams():
    return jsonify({'teams': teams_list, 'venues': venues_list})

@app.route('/api/ml/predict_win', methods=['POST'])
def predict_win():
    try:
        data   = request.json
        team1  = data['team1']
        team2  = data['team2']
        venue  = data['venue']
        toss_w = data['toss_winner']
        toss_d = data['toss_decision']

        t1_enc = int(le_team.transform([team1])[0])
        t2_enc = int(le_team.transform([team2])[0])
        v_enc  = int(le_venue.transform([venue])[0]) if venue in le_venue.classes_ else 0
        d_enc  = int(le_dec.transform([toss_d])[0])
        
        # Advanced Features
        h2h   = h2h_dict.get((team1, team2), 0.5)
        h2h_v = h2h_v_dict.get((team1, team2, venue), 0.5)
        
        v_wr = venue_chase_wr.get(venue, 0.5)
        toss_impact = v_wr if toss_d == 'field' else (1.0 - v_wr)
        
        t1_cf = current_form_dict.get(team1, 0.5)
        t2_cf = current_form_dict.get(team2, 0.5)
        
        proba  = win_model.predict_proba([[t1_enc, t2_enc, v_enc, d_enc, h2h, h2h_v, toss_impact, t1_cf, t2_cf]])[0]
        ml_p1 = float(proba[1])
        ml_p2 = float(proba[0])
        
        return jsonify({
            'team1': team1, 'team2': team2,
            'team1_prob': round(ml_p1 * 100, 1),
            'team2_prob': round(ml_p2 * 100, 1),
            'predicted_winner': team1 if ml_p1 > 0.5 else team2,
            'model_accuracy': win_acc,
            'model': 'Advanced XGBoost'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/ml/clusters')
def player_clusters():
    if cluster_data['batters'] is None:
        return jsonify({'error': 'Clustering unavailable'}), 500
    
    response = {}
    
    # Batters
    b_df = cluster_data['batters']
    b_players = b_df[['batter','strike_rate','avg_runs','total_runs','cluster','role']].copy()
    b_players['total_runs'] = b_players['total_runs'].astype(int)
    b_summary = b_df.groupby('role').agg(
        count=('batter','count'), avg_sr=('strike_rate','mean'), avg_runs=('avg_runs','mean')
    ).reset_index().round(1)
    response['batters'] = {'players': b_players.rename(columns={'batter':'player'}).to_dict('records'), 'summary': b_summary.to_dict('records')}
    
    # Bowlers
    bw_df = cluster_data['bowlers']
    bw_players = bw_df[['bowler','economy','strike_rate','total_wickets','cluster','role']].copy()
    bw_players['total_wickets'] = bw_players['total_wickets'].astype(int)
    bw_summary = bw_df.groupby('role').agg(
        count=('bowler','count'), avg_econ=('economy','mean'), avg_sr=('strike_rate','mean')
    ).reset_index().round(1)
    response['bowlers'] = {'players': bw_players.rename(columns={'bowler':'player'}).to_dict('records'), 'summary': bw_summary.to_dict('records')}
    
    # Allrounders
    ar_df = cluster_data['allrounders']
    ar_players = ar_df[['player','bat_sr','avg_runs','total_runs','economy','total_wickets','cluster','role']].copy()
    ar_players['total_wickets'] = ar_players['total_wickets'].astype(int)
    ar_players['total_runs'] = ar_players['total_runs'].astype(int)
    ar_summary = ar_df.groupby('role').agg(
        count=('player','count'), avg_bat_sr=('bat_sr','mean'), avg_econ=('economy','mean')
    ).reset_index().round(1)
    response['allrounders'] = {'players': ar_players.to_dict('records'), 'summary': ar_summary.to_dict('records')}
    
    response['model'] = 'K-Means Clustering (k=3)'
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)