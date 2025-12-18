"""
This file contains necessary tools and functions for loading the detection model
and for prepping inputs of transaction data.
"""
#-imports-
import joblib
import pandas as pd
from pathlib import Path

MODELPATH = Path("model/best_model.pkl")    # path

def loadModel():
    """
    This function loads the trained model artifact from the disk
    Raises:
        FileNotFoundError: If no model artifact was found
    """
    if not MODELPATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODELPATH} . Run train.py file first.")
    return joblib.load(MODELPATH)

def inputPrep(transaction_dict: dict) -> pd.DataFrame:
    """
    This function converts each single transaction into a
    pandas dataframe to make it compatible with model.
    :param transaction_dict: for Feature name to value mapping
    :return: pd.DataFrame: a singular row DataFrame.
    """
    return pd.DataFrame([transaction_dict])