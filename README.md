# Fraud Detection with SMOTE + Streaming + Flask API - An End-to-End Pipeline

This project, contains an end-to-end fraud detection system. The pipeline trains imbalanced-learning models - both baseline, and SMOTE-enhanced, and selects an optimal class decision threshold based on the validation. Then, it saves a deployable model artifact, a sort of instance of the best model, and services real-time predictions, via a Flask REST API. It also includes a simulator, which streams each record as individiual transaction, to the API.

## Some Highlights:
- **Imbalanced class classifaction** - Through EDA, it is found that there is ~0.17% fraud rate, indicative of a highly skewed target.
- **Threshold Tuning** - The threshold is chosen, through maxmimized optimal F1 scores, rather than the usual default of 0.5.
- **Training Pipeline** - The training pipeline is designed to leak free with train/val/test split with train-only scaling, and train-only SMOTE.
- **Simulates Live Transaction** - the streamer is designed to simulate live transaction, with delays on transaction arrival.
- ** Real-time Inference Decisions** - The Flask REST API produces fraud probabilites and classifies records as fraud or not through threshold-based decisions.

## Tech Stack:
Ensure the below mentioned are downloaded in your project env:
1. Python
2. Pandas
3. numpy
4. scikit-learn
5. imbalanced-learn(SMOTE)
6. Flask - REST API
7. joblib

## Dataset Used:
For this project, I chose to use the **Credit Card Fraud Detection** dataset, named in my files as 'creditcard.csv'. The dataset was found on Kaggle, and can be downloaded following this website: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## In this repo:
train.py              # for training, evaluating and saving artifact of models.
app/api.py            # Flask REST API for inference
app/modelLoader.py    # Helper functions to load artifact of model, and prepare input data from JSON to DataFrame
stream/producer.py    # File for simulator of transaction to be streamed directly to API
notebooks/            # contains any exploratory analysis Jupyter notebooks to analyze, and understand dataset.
model/                # contains the saved artifact model, ignored by git
data/                 # dataset storage and path. Ignored by git
