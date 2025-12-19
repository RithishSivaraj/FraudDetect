# Fraud Detection with SMOTE + Streaming + Flask API - An End-to-End Pipeline

This project contains an end-to-end fraud detection system. The pipeline trains imbalanced-learning models - both baseline, and SMOTE-enhanced, and selects an optimal class decision threshold based on the validation. Then, it saves a deployable model artifact, a sort of instance of the best model, and services real-time predictions, via a Flask REST API. It also includes a simulator, which streams each record as individiual transaction, to the API.

## Some Highlights:
- **Imbalanced class classifaction** - Through EDA, it is found that there is ~0.17% fraud rate, indicative of a highly skewed target.
- **Threshold Tuning** - The threshold is chosen, through maxmimized optimal F1 scores, rather than the usual default of 0.5.
- **Training Pipeline** - The training pipeline is designed to leak free with train/val/test split with train-only scaling, and train-only SMOTE.
- **Simulates Live Transaction** - the streamer is designed to simulate live transaction, with delays on transaction arrival.
- **Real-time Inference Decisions** - The Flask REST API produces fraud probabilites and classifies records as fraud or not through threshold-based decisions.

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
- train.py              # for training, evaluating and saving artifact of models.
- app/api.py            # Flask REST API for inference
- app/modelLoader.py    # Helper functions to load artifact of model, and prepare input data from JSON to DataFrame
- stream/producer.py    # File for simulator of transaction to be streamed directly to API
- notebooks/            # contains any exploratory analysis Jupyter notebooks to analyze, and understand dataset.
- model/                # contains the saved artifact model, ignored by git
- data/                 # dataset storage and path. Ignored by git

## Results:
- Fraud Rate: ~0.17%
- Best Model: Logistic Regression, Baseline (lr_baseline) *More details below*
- Test ROC-AUC: 0.9772945545691546
- Test PR-AUC: 0.758575150867945
- Test Fraud Recall (Class 1): 0.7551
- Validation Threshold: 0.1313

Choosing these metrics as accuracy is misleading with extreme class imbalance, as seen with this dataset, and PR-AUC/Recall reflect performance stats better on fraud cases, where existence of fraud is rare.

## How it all works?
The pipeline follows a few simple but powerful steps:
After confirming dataset specific features and problems you would face, and after preprocessing:
1. train.py -
   - splits the dataset into train/val/test, and scales features.
   - Then apply SMOTE only for training set.
   - Train the different candidate pipelines (Used Logistic regression and Random Forest Classifier for the above)
   - Select the best model using the validation metrics (PR-AUC and F1 score).
   - Save the model into model/best_model.pkl to be loaded into a Flask API later.
2. app/api.py -
   - Loads the artifact of model
   - Only loads once at startup initialization
   - exposes /predict.
3. stream/producer.py -
   - Streams each record from the dataset as transactions into the loaded model in API, in order to simulate live transactions, using an HTTP-based producer.
  
## Setup and Installation
1. Create the virtual environment
   ''' bash
   python -m venv. venv
   source .venv/bin/activate
2. Install the dependent resources
   pip install -r reqs.txt
3. Add the dataset to local directory
   data/raw/creditcard.csv
4. Train the model, and save artifact of best model with
   python train.py
5. Run the API
   python -m app.api
   Note: Keep API running in a terminal and now open a separate terminal
6. Run the producer (in the second terminal, while API is running)
   python stream/producer.py

## General Info
Why is a baseline LR model chosen?
- The model selection is automated based on the validation metrics, particulary PR-AUC and the F1 score. After comparing the 4 candidates, LR and RF under baseline and SMOTE-enhanced, it was found the metrics of baseline LR, outperformed the other models, and was selected as the best model. This may change, depending on different metrics, or a different dataset.

How to prove model is not experiencing majority class baseline problem?
- Due to the extreme class imbalance, the model could always predict false, and still be 99.83% accurate! I ensured that this wasn't the case by sampling the model with a recorded transaction, confirmed as fraud. The model accurately predicted fraud as true!

## Future Improvements to be made:
- Use Dockerfile and docker-compose for one-command deployment
- Better logging and request IDs
- Unit testing for request validation and model loading.

Come back in the future to see more updates!

### Version
1.0
  
