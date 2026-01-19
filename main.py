import pandas as pd
from data_loader import load_data, preprocess_for_ml, get_fixtures
from model import WPLPredictor
import os


def main():
    # File paths
    hist_file = os.path.join("data", "historical_matches.csv")
    fixt_file = "fixtures_2026.csv"

    # 1. Load and Preprocess Training Data
    print("Loading historical data and engineering features...")
    df_hist = load_data(hist_file)
    df_hist, team_to_idx, venue_to_idx = preprocess_for_ml(df_hist)

    # Features: Indices + Rolling Form
    feature_cols = ["team1_idx", "team2_idx", "venue_idx", "team1_form", "team2_form"]
    X = df_hist[feature_cols]
    y = df_hist["label"]

    # 2. Train Model
    print("Training Enhanced Random Forest model...")
    predictor = WPLPredictor()
    # Let's adjust the model to use more trees for better ensemble stability
    predictor.model.set_params(n_estimators=200, min_samples_split=5)
    predictor.train(X, y, team_to_idx, venue_to_idx)
    predictor.save_model()

    # 3. Load 2026 Fixtures
    print("Loading and preparing upcoming fixtures...")
    df_fixt = get_fixtures(fixt_file, team_to_idx, venue_to_idx, df_hist)
    X_fixt = df_fixt[feature_cols]

    # 4. Predict
    print("Generating predictions with momentum awareness...")
    preds, probs = predictor.predict(X_fixt)

    # 5. Output Results
    results = []
    for i, row in df_fixt.iterrows():
        winner = row["team1"] if preds[i] == 1 else row["team2"]
        confidence = probs[i][1] if preds[i] == 1 else probs[i][0]

        results.append(
            {
                "Date": row["date"].strftime("%Y-%m-%d"),
                "Venue": row["venue"],
                "Team 1": row["team1"],
                "Team 2": row["team2"],
                "T1 Form": f"{row['team1_form']:.2f}",
                "T2 Form": f"{row['team2_form']:.2f}",
                "Predicted Winner": winner,
                "Confidence": f"{confidence:.2%}",
            }
        )

    results_df = pd.DataFrame(results)

    # Filter for future matches only (after Jan 19, 2026) for the user's focus
    # But for full file we keep all.
    print("\n--- UPDATED WPL 2026 PREDICTIONS (Toss-Adjusted Features) ---")
    print(results_df.to_string(index=False))

    results_df.to_csv("wpl_2026_predictions.csv", index=False)
    print("\nPredictions saved to wpl_2026_predictions.csv")


if __name__ == "__main__":
    main()
