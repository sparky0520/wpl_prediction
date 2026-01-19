import pandas as pd
import numpy as np


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def preprocess_for_ml(df):
    # Encode teams and venues
    teams = sorted(list(set(df["team1"].unique()) | set(df["team2"].unique())))
    venues = sorted(list(df["venue"].unique()))

    team_to_idx = {team: i for i, team in enumerate(teams)}
    venue_to_idx = {venue: i for i, venue in enumerate(venues)}

    df["team1_idx"] = df["team1"].map(team_to_idx)
    df["team2_idx"] = df["team2"].map(team_to_idx)
    df["venue_idx"] = df["venue"].map(venue_to_idx)

    # 1 if team1 wins, 0 if team2 wins
    df["label"] = (df["winner"] == df["team1"]).astype(int)

    return df, team_to_idx, venue_to_idx


def get_fixtures(filepath, team_to_idx, venue_to_idx):
    fixtures = pd.read_csv(filepath)

    # Handle potentially new venues in 2026
    current_venues = list(venue_to_idx.keys())
    for venue in fixtures["venue"].unique():
        if venue not in venue_to_idx:
            venue_to_idx[venue] = len(venue_to_idx)

    fixtures["team1_idx"] = fixtures["team1"].map(team_to_idx)
    fixtures["team2_idx"] = fixtures["team2"].map(team_to_idx)
    fixtures["venue_idx"] = fixtures["venue"].map(venue_to_idx)

    return fixtures
