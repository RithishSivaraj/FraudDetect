import joblib
import pandas as pd
from pathlib import Path

MODELPATH = Path("model/best_model.pkl")

def loadModel():
    if not MODELPATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODELPATH} . Run train.py file first.")
    return joblib.load(MODELPATH)

def inputPrep(transaction_dict: dict) -> pd.DataFrame:
    return pd.DataFrame([transaction_dict])