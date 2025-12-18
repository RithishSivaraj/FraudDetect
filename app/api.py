from flask import Flask, request, jsonify
from app.modelLoader import loadModel, inputPrep

app = Flask(__name__)

artifact = loadModel()
pipeline = artifact["pipeline"]
threshold = float(artifact["threshold"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"status": "error", "message": "pipeline is None"}), 500

    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "payload is None"}), 400

    try:
        X = inputPrep(payload)
        prob = float(pipeline.predict_prob(X)[:, 1][0])
        isFraud = bool(prob >= threshold)
        return jsonify({
            "fraud_probability": prob,
            "threshold": threshold,
            "isFraud": isFraud
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
