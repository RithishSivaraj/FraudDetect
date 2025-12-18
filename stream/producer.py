"""
This file will simulate real-time transactions done by extracting each row
from the dataset, and inputting them into the fraud detection REST API, one by one.
Each row is read as a separate transaction and checked for fraud.
"""
#-imports-
import time
import requests
import pandas as pd
from pathlib import Path

TESTPATH = Path("data/raw/creditcard.csv") # path for dataset

def streamer(api_url= "http://localhost:5001", delay=0.5, limit=100):
    """
    Streams the dataset rows as separate transactions to the prediction REST API.
    :param api_url: the base URL of the REST API, running.
    :param delay: The set time delay in seconds between each request.
    :param limit: maximum number of rows to read (transactions)
    """

    df = pd.read_csv(TESTPATH) # loading the dataset

    # removing the feature label before sending it to the API
    featuredf= df.drop(columns=["Class"])

    for i in range(min(limit, len(featuredf))):
        row = featuredf.iloc[i].to_dict()   # converting a single row into JSON style format.
        try:
            r = requests.post(f"{api_url}/predict", json=row, timeout=5)    # sending the transaction to prediction endpoint.
            print(f"[{i}] status={r.status_code} resp={r.json()}")  # logging responses
        except Exception as e:
            print(f"[{i}] error: {e}")      # handling any errros

        time.sleep(delay)   # Pausing the simulator in order to test real-time transactions.

if __name__ == "__main__":
    streamer()  # run the streamer as script.
