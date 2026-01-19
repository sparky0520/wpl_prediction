from sklearn.ensemble import RandomForestClassifier
import pickle
import os


class WPLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.team_to_idx = None
        self.venue_to_idx = None
        self.idx_to_team = None

    def train(self, X, y, team_to_idx, venue_to_idx):
        self.team_to_idx = team_to_idx
        self.venue_to_idx = venue_to_idx
        self.idx_to_team = {i: team for team, i in team_to_idx.items()}
        self.model.fit(X, y)

    def predict(self, X_fixt):
        probs = self.model.predict_proba(X_fixt)
        preds = self.model.predict(X_fixt)
        return preds, probs

    def save_model(self, path="wpl_model.pkl"):
        data = {
            "model": self.model,
            "team_to_idx": self.team_to_idx,
            "venue_to_idx": self.venue_to_idx,
            "idx_to_team": self.idx_to_team,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, path="wpl_model.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.team_to_idx = data["team_to_idx"]
                self.venue_to_idx = data["venue_to_idx"]
                self.idx_to_team = data["idx_to_team"]
            return True
        return False
