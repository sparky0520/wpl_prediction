print("Hello form test")
import pandas as pd

print("Pandas imported")
try:
    df = pd.read_csv("data/historical_matches.csv")
    print(f"Loaded {len(df)} rows")
except Exception as e:
    print(f"Error: {e}")
