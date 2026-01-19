import pandas as pd
import numpy as np


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def calculate_rolling_form(df, team_name, current_date, window=5):
    # Get all matches for this team before the current date
    team_matches = df[
        ((df["team1"] == team_name) | (df["team2"] == team_name))
        & (df["date"] < current_date)
    ]
    if len(team_matches) == 0:
        return 0.5  # Neutral starting point

    last_n = team_matches.tail(window)
    wins = len(last_n[last_n["winner"] == team_name])
    return wins / len(last_n)


def preprocess_for_ml(df):
    # Encode teams and venues
    teams = sorted(list(set(df["team1"].unique()) | set(df["team2"].unique())))
    venues = sorted(list(df["venue"].unique()))

    team_to_idx = {team: i for i, team in enumerate(teams)}
    venue_to_idx = {venue: i for i, venue in enumerate(venues)}

    # Calculate Features: Rolling Form
    df["team1_form"] = df.apply(
        lambda x: calculate_rolling_form(df, x["team1"], x["date"]), axis=1
    )
    df["team2_form"] = df.apply(
        lambda x: calculate_rolling_form(df, x["team2"], x["date"]), axis=1
    )

    # Calculate Features: Venue Bias (Simplified for Pre-match)
    # Since we don't know the toss yet, we just provide the venue's overall home/away win rates
    df["team1_idx"] = df["team1"].map(team_to_idx)
    df["team2_idx"] = df["team2"].map(team_to_idx)
    df["venue_idx"] = df["venue"].map(venue_to_idx)

    # Label: 1 if team1 wins, 0 if team2 wins
    df["label"] = (df["winner"] == df["team1"]).astype(int)

    return df, team_to_idx, venue_to_idx


def get_fixtures(filepath, team_to_idx, venue_to_idx, full_hist_df):
    fixtures = pd.read_csv(filepath)
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    # Handle potentially new venues
    for venue in fixtures["venue"].unique():
        if venue not in venue_to_idx:
            venue_to_idx[venue] = len(venue_to_idx)

    # Calculate form based on latest status in full_hist_df
    fixtures["team1_form"] = fixtures.apply(
        lambda x: calculate_rolling_form(full_hist_df, x["team1"], x["date"]), axis=1
    )
    fixtures["team2_form"] = fixtures.apply(
        lambda x: calculate_rolling_form(full_hist_df, x["team2"], x["date"]), axis=1
    )

    fixtures["team1_idx"] = fixtures["team1"].map(team_to_idx)
    fixtures["team2_idx"] = fixtures["team2"].map(team_to_idx)
    fixtures["venue_idx"] = fixtures["venue"].map(venue_to_idx)

    return fixtures
