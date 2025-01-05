from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("models/credit_risk_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data["features"]])
    return jsonify({"credit_risk": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
