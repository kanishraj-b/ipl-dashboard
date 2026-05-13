from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ══════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════
print("Loading IPL data...")
df = pd.read_csv("IPL.zip", low_memory=False)
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
#  ML MODEL 1 — Score Predictor (Linear Regression)
# ══════════════════════════════════════════════════════
print("Training Score Predictor (Linear Regression)...")
try:
    pp_df = df[df['over'] <= 6]
    pp_runs_per_match    = pp_df.groupby('match_id')['runs_batter'].sum()
    pp_wkts_per_match    = pp_df.groupby('match_id')['bowler_wicket'].sum()
    total_runs_per_match = df.groupby('match_id')['runs_total'].sum()

    score_data = pd.DataFrame({
        'pp_runs':    pp_runs_per_match,
        'pp_wickets': pp_wkts_per_match,
        'total':      total_runs_per_match
    }).dropna()
  
    X_sc = score_data[['pp_runs', 'pp_wickets']]
    y_sc = score_data['total']
    score_model = LinearRegression()
    score_model.fit(X_sc, y_sc)
    score_r2 = round(score_model.score(X_sc, y_sc), 3)
    avg_pp_runs = round(pp_runs_per_match.mean(), 1)
    print (f"   R² = {score_r2}")
except Exception as e:
    print(f"   Error: {e}")
    score_model = None
    score_r2 = 0
    avg_pp_runs = 50

# ══════════════════════════════════════════════════════
#  ML MODEL 2 — Win Predictor (Decision Tree)
# ══════════════════════════════════════════════════════
print("Training Win Predictor (Decision Tree)...")
win_model = None
win_acc = 0
teams_list = []
venues_list = []
le_team = LabelEncoder()
le_venue = LabelEncoder()
le_dec = LabelEncoder()
win_counts_dict = {}
total_games_dict = {}
recent_win_counts = {}
recent_total_games = {}

try:
    m = matches.dropna(subset=['toss_winner', 'toss_decision', 'venue',
                                'batting_team', 'bowling_team', 'match_won_by']).copy()

    all_teams = pd.concat([m['batting_team'], m['bowling_team'], m['toss_winner']]).unique()
    le_team.fit(all_teams)
    le_venue.fit(m['venue'])
    le_dec.fit(m['toss_decision'])

    win_counts_dict = m['match_won_by'].value_counts().to_dict()
    for t in all_teams:
        total_games_dict[t] = int((m['batting_team'] == t).sum() + (m['bowling_team'] == t).sum())

    # --- RECENT FORM DICTS (last 3 seasons) ---
    if 'season' in m.columns:
        valid_seasons = sorted(m['season'].dropna().unique())
        last_3 = valid_seasons[-3:] if len(valid_seasons) >= 3 else valid_seasons
        m_recent = m[m['season'].isin(last_3)]
        recent_win_counts = m_recent['match_won_by'].value_counts().to_dict()
        for t in all_teams:
            recent_total_games[t] = int((m_recent['batting_team'] == t).sum() + (m_recent['bowling_team'] == t).sum())

    m['t1_enc']  = le_team.transform(m['batting_team'])
    m['t2_enc']  = le_team.transform(m['bowling_team'])
    m['v_enc']   = le_venue.transform(m['venue'])
    m['d_enc']   = le_dec.transform(m['toss_decision'])
    m['t1_wr']   = m['batting_team'].map(lambda t: win_counts_dict.get(t, 0) / max(total_games_dict.get(t, 1), 1))
    m['t2_wr']   = m['bowling_team'].map(lambda t: win_counts_dict.get(t, 0) / max(total_games_dict.get(t, 1), 1))
    m['target']  = (m['match_won_by'] == m['batting_team']).astype(int)

    feats = ['t1_enc', 't2_enc', 'v_enc', 'd_enc', 't1_wr', 't2_wr']
    X_w, y_w = m[feats], m['target']
    Xtr, Xte, ytr, yte = train_test_split(X_w, y_w, test_size=0.3, random_state=42)
    win_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    win_model.fit(Xtr, ytr)
    win_acc = round(accuracy_score(yte, win_model.predict(Xte)) * 100, 1)
    teams_list  = sorted(all_teams.tolist())
    venues_list = sorted(m['venue'].unique().tolist())
    print(f"   Accuracy = {win_acc}%")
except Exception as e:
    print(f"   Error: {e}")

# ══════════════════════════════════════════════════════
#  ML MODEL 3 — Player Clustering (K-Means)
# ══════════════════════════════════════════════════════
print("Training Player Clusters (K-Means)...")
cluster_data = {'batters': None, 'bowlers': None, 'allrounders': None}
try:
    # --- BATTERS ---
    entry_overs = df.groupby(['match_id', 'batter'])['over'].min().groupby('batter').mean().reset_index()
    entry_overs.rename(columns={'over': 'avg_entry_over'}, inplace=True)

    bs = df.groupby('batter').agg(
        total_runs=('runs_batter', 'sum'),
        total_balls=('balls_faced', 'sum'),
        matches=('match_id', 'nunique')
    ).reset_index()
    bs = bs.merge(entry_overs, on='batter', how='left')
    bs = bs[bs['total_balls'] >= 200].copy()
    bs['strike_rate'] = (bs['total_runs'] / bs['total_balls'] * 100).round(2)
    bs['avg_runs']    = (bs['total_runs'] / bs['matches']).round(2)

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
    v = df[df['valid_ball'] == 1]
    
    # Calculate middle overs percentage to infer spin/pace
    mid_mask = (df['over'] > 6) & (df['over'] < 16)
    mid_counts = df[mid_mask].groupby('bowler').size().reset_index(name='is_mid')
    
    bowls = v.groupby('bowler').agg(
        total_runs=('runs_bowler', 'sum'),
        total_balls=('valid_ball', 'sum'),
        total_wickets=('bowler_wicket', 'sum')
    ).reset_index()
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
    
    cluster_stats = bowls.groupby('cluster').agg(
        avg_mid_pct=('mid_pct', 'mean'),
        avg_econ=('economy', 'mean'),
        avg_sr=('strike_rate', 'mean')
    ).reset_index()
    
    cluster_stats = cluster_stats.sort_values('avg_mid_pct', ascending=False)
    spinners = cluster_stats.head(2).sort_values('avg_econ')
    pacers = cluster_stats.tail(2).sort_values('avg_sr')
    
    labels_map_bw = {
        spinners.iloc[0]['cluster']: '🌀 Off Spinner',
        spinners.iloc[1]['cluster']: '🪄 Leg Spinner',
        pacers.iloc[0]['cluster']: '🚀 Pacer',
        pacers.iloc[1]['cluster']: '🎯 Medium Pacer'
    }
    
    bowls['role'] = bowls['cluster'].map(labels_map_bw)
    cluster_data['bowlers'] = bowls

    # --- ALLROUNDERS ---
    ar = pd.merge(bs[['batter', 'total_runs', 'strike_rate', 'avg_runs']],
                  bowls[['bowler', 'total_wickets', 'economy', 'strike_rate']],
                  left_on='batter', right_on='bowler', how='inner')
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

    print(f"   Clustered {len(bs)} batters, {len(bowls)} bowlers, {len(ar)} allrounders")
except Exception as e:
    print(f"   Error: {e}")

print("OK: All models ready!\n")

# ══════════════════════════════════════════════════════
#  ANALYSIS ROUTES
# ══════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/team_wins')
def team_wins():
    data = matches['match_won_by'].value_counts().head(10)
    return jsonify({'teams': data.index.tolist(), 'wins': data.values.tolist()})

@app.route('/api/toss_effect')
def toss_effect():
    m2 = matches.dropna(subset=['toss_winner', 'match_won_by']).copy()
    m2['tw'] = m2['toss_winner'] == m2['match_won_by']
    c = m2['tw'].value_counts()
    return jsonify({'labels': ['Toss Winner Won', 'Toss Winner Lost'],
                    'values': [int(c.get(True, 0)), int(c.get(False, 0))]})

@app.route('/api/toss_decision')
def toss_decision():
    data = matches['toss_decision'].value_counts()
    return jsonify({'decisions': data.index.tolist(), 'counts': data.values.tolist()})

@app.route('/api/top_batters')
def top_batters():
    s = df.groupby('batter').agg(total_runs=('runs_batter','sum'), total_balls=('balls_faced','sum')).reset_index()
    s['strike_rate'] = (s['total_runs'] / s['total_balls'] * 100).round(2)
    top = s[s['total_balls'] >= 500].sort_values('strike_rate', ascending=False).head(10)
    return jsonify({'batters': top['batter'].tolist(), 'strike_rates': top['strike_rate'].tolist(), 'total_runs': top['total_runs'].tolist()})

@app.route('/api/top_bowlers')
def top_bowlers():
    v = df[df['valid_ball'] == 1]
    s = v.groupby('bowler').agg(total_runs=('runs_bowler','sum'), total_balls=('valid_ball','sum'), total_wickets=('bowler_wicket','sum')).reset_index()
    s['economy'] = (s['total_runs'] / s['total_balls'] * 6).round(2)
    top = s[s['total_balls'] >= 1000].sort_values('economy').head(10)
    return jsonify({'bowlers': top['bowler'].tolist(), 'economy': top['economy'].tolist(), 'wickets': top['total_wickets'].tolist()})

@app.route('/api/phase_runs')
def phase_runs():
    pp = df[df['over'] <= 6].groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(8)
    dt = df[df['over'] >= 17].groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(8)
    return jsonify({'pp_batters': pp.index.tolist(), 'pp_runs': pp.values.tolist(),
                    'dt_batters': dt.index.tolist(), 'dt_runs': dt.values.tolist()})

@app.route('/api/dismissals')
def dismissals():
    data = df[df['wicket_kind'].notna()]['wicket_kind'].value_counts()
    return jsonify({'types': data.index.tolist(), 'counts': data.values.tolist()})

@app.route('/api/season_scores')
def season_scores():
    ts = df.groupby(['season','match_id','batting_team'])['runs_total'].sum().reset_index()
    avg = ts.groupby('season')['runs_total'].mean().round(1)
    return jsonify({'seasons': [str(s) for s in avg.index.tolist()], 'avg_scores': avg.values.tolist()})

@app.route('/api/boundaries')
def boundaries():
    fours = df[df['runs_batter'] == 4].groupby('batter').size().reset_index(name='fours')
    sixes = df[df['runs_batter'] == 6].groupby('batter').size().reset_index(name='sixes')
    top_4s = fours.sort_values('fours', ascending=False).head(10)
    top_6s = sixes.sort_values('sixes', ascending=False).head(10)
    return jsonify({
        'top_4s': {'batters': top_4s['batter'].tolist(), 'count': top_4s['fours'].tolist()},
        'top_6s': {'batters': top_6s['batter'].tolist(), 'count': top_6s['sixes'].tolist()}
    })

@app.route('/api/orange_cap')
def orange_cap():
    col = 'season' if 'season' in df.columns else 'year'
    season_runs = df.groupby([col, 'batter'])['runs_batter'].sum().reset_index()
    idx = season_runs.groupby(col)['runs_batter'].idxmax()
    orange_caps = season_runs.loc[idx].sort_values(col)
    return jsonify({
        'seasons': orange_caps[col].astype(str).tolist(),
        'batters': orange_caps['batter'].tolist(),
        'runs': orange_caps['runs_batter'].tolist()
    })

@app.route('/api/phase_wickets')
def phase_wickets():
    pp = df[df['over'] <= 6].groupby('bowler')['bowler_wicket'].sum().sort_values(ascending=False).head(8)
    dt = df[df['over'] >= 17].groupby('bowler')['bowler_wicket'].sum().sort_values(ascending=False).head(8)
    return jsonify({'pp_bowlers': pp.index.tolist(), 'pp_wickets': pp.values.tolist(),
                    'dt_bowlers': dt.index.tolist(), 'dt_wickets': dt.values.tolist()})

@app.route('/api/purple_cap')
def purple_cap():
    col = 'season' if 'season' in df.columns else 'year'
    season_wkts = df.groupby([col, 'bowler'])['bowler_wicket'].sum().reset_index()
    idx = season_wkts.groupby(col)['bowler_wicket'].idxmax()
    purple_caps = season_wkts.loc[idx].sort_values(col)
    return jsonify({
        'seasons': purple_caps[col].astype(str).tolist(),
        'bowlers': purple_caps['bowler'].tolist(),
        'wickets': purple_caps['bowler_wicket'].tolist()
    })

# ══════════════════════════════════════════════════════
#  ML ROUTES
# ══════════════════════════════════════════════════════

@app.route('/api/ml/predict_score', methods=['POST'])
def predict_score():
    try:
        data      = request.json
        pp_runs   = float(data['powerplay_runs'])
        pp_wkts   = float(data['powerplay_wickets'])
        predicted = int(max(60, min(300, round(score_model.predict([[pp_runs, pp_wkts]])[0])))*(1-pp_wkts*.1))
        return jsonify({
            'predicted_score': predicted,
            'r2_score': score_r2,
            'model': 'Linear Regression',
            'avg_powerplay': avg_pp_runs,
            'insight': 'above average powerplay' if pp_runs > avg_pp_runs else 'below average powerplay'
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
        t1_wr  = win_counts_dict.get(team1, 0) / max(total_games_dict.get(team1, 1), 1)
        t2_wr  = win_counts_dict.get(team2, 0) / max(total_games_dict.get(team2, 1), 1)
        proba  = win_model.predict_proba([[t1_enc, t2_enc, v_enc, d_enc, t1_wr, t2_wr]])[0]
        ml_p1 = float(proba[1])
        ml_p2 = float(proba[0])
        
        t1_rw = recent_win_counts.get(team1, 0) / max(recent_total_games.get(team1, 1), 1)
        t2_rw = recent_win_counts.get(team2, 0) / max(recent_total_games.get(team2, 1), 1)
        
        if t1_rw + t2_rw == 0:
            rec_p1 = 0.5
            rec_p2 = 0.5
        else:
            rec_p1 = t1_rw / (t1_rw + t2_rw)
            rec_p2 = t2_rw / (t1_rw + t2_rw)
            
        final_p1 = 0.25 * ml_p1 + 0.75 * rec_p1
        final_p2 = 0.25 * ml_p2 + 0.75 * rec_p2

        return jsonify({
            'team1': team1, 'team2': team2,
            'team1_prob': round(final_p1 * 100, 1),
            'team2_prob': round(final_p2 * 100, 1),
            'predicted_winner': team1 if final_p1 > 0.5 else team2,
            'model_accuracy': win_acc,
            'model': 'XGBoost (25%) + Recent Form (75%)'
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
    app.run(debug=True)