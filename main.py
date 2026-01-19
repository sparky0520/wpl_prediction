import pandas as pd
from data_loader import load_data, preprocess_for_ml, get_fixtures
from model import WPLPredictor
import os


def main():
    # File paths
    hist_file = os.path.join("data", "historical_matches.csv")
    fixt_file = "fixtures_2026.csv"

    # 1. Load and Preprocess Training Data
    print("Loading historical data...")
    df_hist = load_data(hist_file)
    df_hist, team_to_idx, venue_to_idx = preprocess_for_ml(df_hist)

    X = df_hist[["team1_idx", "team2_idx", "venue_idx"]]
    y = df_hist["label"]

    # 2. Train Model
    print("Training model...")
    predictor = WPLPredictor()
    predictor.train(X, y, team_to_idx, venue_to_idx)
    predictor.save_model()

    # 3. Load 2026 Fixtures
    print("Loading 2026 fixtures...")
    df_fixt = get_fixtures(fixt_file, team_to_idx, venue_to_idx)
    X_fixt = df_fixt[["team1_idx", "team2_idx", "venue_idx"]]

    # 4. Predict
    print("Generating predictions...")
    preds, probs = predictor.predict(X_fixt)

    # 5. Output Results
    results = []
    for i, row in df_fixt.iterrows():
        # pred 1 means team1 wins, 0 means team2 wins
        winner = row["team1"] if preds[i] == 1 else row["team2"]
        confidence = probs[i][1] if preds[i] == 1 else probs[i][0]

        results.append(
            {
                "Date": row["date"],
                "Venue": row["venue"],
                "Team 1": row["team1"],
                "Team 2": row["team2"],
                "Predicted Winner": winner,
                "Confidence": f"{confidence:.2%}",
            }
        )

    results_df = pd.DataFrame(results)
    print("\n--- WPL 2026 PREDICTIONS ---")
    print(results_df.to_string(index=False))

    results_df.to_csv("wpl_2026_predictions.csv", index=False)
    print("\nPredictions saved to wpl_2026_predictions.csv")


if __name__ == "__main__":
    main()
