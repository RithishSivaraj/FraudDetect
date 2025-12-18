# Rithish Sivaraj

# Phase 1: Data Preprocessing:

# -imports-
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeLine


#-loading data-
RAWDATAPATH = Path("data/raw/creditcard.csv")
MODELDIRPATH = Path("model")
MODELPATH = MODELDIRPATH / "best_model.pkl"
RANDOM_STATE = 42

def dataloader() -> pd.DataFrame:
    if not RAWDATAPATH.exists():
        print(f"Dataset not found: {RAWDATAPATH}")
    df = pd.read_csv(RAWDATAPATH)
    if "Class" not in df.columns:
        print("The target column 'Class' is not found within the dataset.")
    return df

#-splitting data-
def dataSplit(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # creating the first split, for train and val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # creating the second split, train vs val - from trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
                y_trainval,
                test_size=0.2,
                stratify=y_trainval,
                random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def probsEval(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    crep = classification_report(y_true, y_pred, digits=4, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    return cm, crep, roc, prauc

# to choose a threshold which maximizes the F1 on the validation set.
def threshold_chooser(y_val, val_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    # thresholds has a length of (len(precisions)-1). Align:
    f1Scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
    bestIDX = int(np.argmax(f1Scores))
    bestThreshold = float(thresholds[bestIDX])
    return bestThreshold, float(precisions[bestIDX]), float(recalls[bestIDX]), float(f1Scores[bestIDX])

# here we are training a baseline model without SMOTE, and one with smote, to compare the two.
def trainCompare(X_train, y_train, X_val, y_val):

    lr = LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)


    candidates = []

    # training without SMOTE, so baseline pipelines consisting of only the scaler and models:
    candidates.append(("lr_baseline", ImbPipeLine([("scaler", StandardScaler()), ("clf", lr)])))

    candidates.append(("rf_baseline", ImbPipeLine([("scaler", StandardScaler()), ("clf", rf)])))

    # Now training with SMOTE pipelines which consist of scaler, SMOTE and the models.
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

    candidates.append(("lr_smote", ImbPipeLine([("scaler", StandardScaler()), ("smote", smote), ("clf", rf)])))

    candidates.append(("rf_smote", ImbPipeLine([("scaler", StandardScaler()), ("smote", smote), ("clf", rf)])))

    results = []

    for name, pipe in candidates:
        print(f"\n-----Training: {name} -----")
        pipe.fit(X_train, y_train)

        val_probs = pipe.predict_proba(X_val)[:, 1]
        best_threshold, p, r, f1 = threshold_chooser(y_val, val_probs)
        cm, crep, roc, prauc = probsEval(y_val, val_probs, best_threshold)

        print(f"Best threshold on VAL (max F1): {best_threshold:.4f}")
        print(f"VAL Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
        print(f"VAL ROC-AUC={roc:.4f} PR-AUC={prauc:.4f}")
        print("Confusion Matrix (VAL):\n", cm)
        print("Classification Report (VAL):\n", crep)

        results.append({
            "name": name,
            "pipeline": pipe,
            "val_threshold": best_threshold,
            "val_f1": f1,
            "val_precision": p,
            "val_recall": r,
            "val_roc_auc": roc,
            "val_pr_auc": prauc
        })


        sortedResults = sorted(results, key=lambda d: (d["val_pr_auc"], d["val_f1"]), reverse=True)
        best = sortedResults[0]

        print("")
        print("Selected Best Model: ", best["name"])
        print(" (by PR-AUC, then by F1 val)")

        return best

def testEval_Final(best, X_test, y_test):
    pipe = best['pipeline']
    threshold = best['val_threshold']

    test_probs = pipe.predict_proba(X_test)[:, 1]
    cm, crep, roc, prauc = probsEval(y_test, test_probs, threshold=threshold)

    print("")
    print("----Final Test Eval Results----")
    print(f"confusion matrix: {cm}")
    print(f"classification report: {crep}")
    print(f"ROC AUC: {roc}")
    print(f"PR-AUC: {prauc}")
    print(f"upon using threshold from val: {threshold:.4}")


def artiSave(best):
    MODELDIRPATH.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": best["pipeline"],
        "val_threshold": best["val_threshold"],
        "meta": {
            "name": best["name"],
            "random_state": RANDOM_STATE
        }
    }

    joblib.dump(artifact, MODELDIRPATH / f"{best['name']}.joblib")
    print(f"Model saved. Artifact to -> {MODELDIRPATH}")

def main():
    df = dataloader()
    print(f"Data loaded. Shape: {df.shape}")

    print()
    print("Class dist. Overall:")
    print(df["Class"].value_counts(normalize=True))

    X_train, X_val, X_test, y_train, y_val, y_test = dataSplit(df)

    print()
    print("Split Sizes:")
    print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)
    print()
    print("Fraud Ratio Rate:")
    print("Train:", y_train.mean(), "Validation:", y_val.mean(), "Test:", y_test.mean())

    best = trainCompare(X_train, y_train, X_val, y_val)
    testEval_Final(best, X_test, y_test)
    artiSave(best)

if __name__ == "__main__":
    main()