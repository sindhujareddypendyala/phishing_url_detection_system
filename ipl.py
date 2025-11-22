import streamlit as st
import random

# --- Page Setup ---
st.set_page_config(
    page_title="Cricket Match Predictor",
    page_icon="🏏",
    layout="wide"
)

# --- Custom CSS for cricket theme ---
st.markdown(
    """
    <style>
    /* Background image */
    [data-testid="stAppViewContainer"] {
        background-image: url('https://images.unsplash.com/photo-1560464024-54b1d1f87725?auto=format&fit=crop&w=1470&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        filter: brightness(0.8);
    }

    /* Overlay for readability */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.4);
        z-index: -1;
    }

    .main-title {
        color: #FFD700;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 40px;
    }

    .stButton>button {
        background-color: #FF4500;
        color: white;
        height: 50px;
        width: 200px;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<h1 class="main-title">🏏 Cricket Match Predictor 🏏</h1>', unsafe_allow_html=True)

# --- IPL Teams ---
teams = ["CSK", "MI", "RCB", "KKR", "SRH", "DC", "RR", "PBKS","LSG","GT"]

# --- Functions ---
def predict_final_score(current_score, rr, wickets, overs_played, total_overs=20):
    overs_left = total_overs - overs_played
    predicted_additional_runs = rr * overs_left * ((10 - wickets)/10)
    return round(current_score + predicted_additional_runs)

def calculate_win_probability(batting_team, bowling_team, toss_winner, toss_choice,
                              score_bat, rr_bat, wickets_bat, overs_bat,
                              score_bowl, rr_bowl, wickets_bowl, overs_bowl,
                              total_overs=20):
    final_score_bat = predict_final_score(score_bat, rr_bat, wickets_bat, overs_bat, total_overs)
    final_score_bowl = predict_final_score(score_bowl, rr_bowl, wickets_bowl, overs_bowl, total_overs)

    # Toss advantage
    if toss_winner == batting_team and toss_choice == "bat":
        final_score_bat += 5
    elif toss_winner == bowling_team and toss_choice == "field":
        final_score_bowl += 5

    # Performance scores
    perf_bat = final_score_bat * ((10 - wickets_bat)/10)
    perf_bowl = final_score_bowl * ((10 - wickets_bowl)/10)

    total_perf = perf_bat + perf_bowl
    win_prob_bat = round((perf_bat / total_perf) * 100)
    win_prob_bowl = 100 - win_prob_bat

    predicted_winner = batting_team if win_prob_bat > win_prob_bowl else bowling_team

    return predicted_winner, final_score_bat, final_score_bowl, win_prob_bat, win_prob_bowl

# --- Sidebar Inputs ---
st.sidebar.header("Match Setup")
team1 = st.sidebar.selectbox("Select Team 1", teams)
team2 = st.sidebar.selectbox("Select Team 2", [t for t in teams if t != team1])
toss_winner = st.sidebar.selectbox("Toss Winner", [team1, team2])
toss_choice = st.sidebar.radio("Toss Choice", ["bat", "field"])

# Decide batting and bowling team
if toss_choice == "bat":
    batting_team = toss_winner
    bowling_team = team2 if toss_winner == team1 else team1
else:
    bowling_team = toss_winner
    batting_team = team2 if toss_winner == team1 else team1

st.write(f"**Batting Team:** {batting_team}")
st.write(f"**Bowling Team:** {bowling_team}")

st.sidebar.header("Batting Team Stats")
score_bat = st.sidebar.number_input(f"{batting_team} Current Score", min_value=0, value=50)
rr_bat = st.sidebar.number_input(f"{batting_team} Current Run Rate", min_value=0.0, value=6.0)
wickets_bat = st.sidebar.number_input(f"{batting_team} Wickets Fallen", min_value=0, max_value=10, value=2)
overs_bat = st.sidebar.number_input(f"{batting_team} Overs Played", min_value=0.0, max_value=20.0, value=10.0)

st.sidebar.header("Bowling Team Stats")
score_bowl = st.sidebar.number_input(f"{bowling_team} Current Score", min_value=0, value=50)
rr_bowl = st.sidebar.number_input(f"{bowling_team} Current Run Rate", min_value=0.0, value=6.0)
wickets_bowl = st.sidebar.number_input(f"{bowling_team} Wickets Fallen", min_value=0, max_value=10, value=2)
overs_bowl = st.sidebar.number_input(f"{bowling_team} Overs Played", min_value=0.0, max_value=20.0, value=10.0)

# --- Predict Button ---
if st.button("Predict Match"):
    winner, final_score_bat, final_score_bowl, win_prob_bat, win_prob_bowl = calculate_win_probability(
        batting_team, bowling_team, toss_winner, toss_choice,
        score_bat, rr_bat, wickets_bat, overs_bat,
        score_bowl, rr_bowl, wickets_bowl, overs_bowl
    )

    st.subheader("🏏 Predicted Final Scores")
    st.write(f"{batting_team}: {final_score_bat}")
    st.write(f"{bowling_team}: {final_score_bowl}")

    st.subheader("💯 Win Probability")
    st.write(f"{batting_team}: {win_prob_bat}%")
    st.write(f"{bowling_team}: {win_prob_bowl}%")

    st.subheader("🏆 Predicted Winner")
    st.success(f"{winner} is likely to win! 🎉")
    st.balloons()
