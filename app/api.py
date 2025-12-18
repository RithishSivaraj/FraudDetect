"""
This file exposes a REST API in order to check real-time fraud prediction.
It loads the trained model artifact at the startup and serves the predictions
through HTTP endpoints.
"""

# -imports-
from flask import Flask, request, jsonify
from app.modelLoader import loadModel, inputPrep

app = Flask(__name__)   # Flask application initialization

# loading trained model artifact at the startup of application
# avoids loading at every single request
artifact = loadModel()
pipeline = artifact["pipeline"]
threshold = float(artifact["threshold"])
@app.route("/health", methods=["GET"])
def health():
    """
    Health check for endpoint, to check that service is still alive.
    """
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Finds the fraud probability, through a transaction payload and threshold
    based classification.
    :return: fraud probability prediction
    """
    if pipeline is None:    # safety check
        return jsonify({"status": "error", "message": "pipeline is None"}), 500

    payload = request.get_json(silent=True)     # parsing the JSON payload
    if payload is None or not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "payload is None"}), 400

    try:
        X = inputPrep(payload)  # prepping input for model
        prob = float(pipeline.predict_proba(X)[:, 1][0])    #p predicting probability of fraud. Class -> 1
        isFraud = bool(prob >= threshold)   # applying decision threshold, retrieved from validation stage.
        return jsonify({
            "fraud_probability": prob,
            "threshold": threshold,
            "isFraud": isFraud
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400      # return any inference time errors.


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # running development server.
    # Note: I set port to 5001 since my port 5000 was busy, may have to change ports depending on usage.
