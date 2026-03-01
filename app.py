"""
Cardiovascular Disease Prediction API
======================================
Matches EXACTLY the preprocessing pipeline from week_2.ipynb.

Notebook feature engineering recap:
  1. age_years = int(age_days / 365)       (or accept age_years directly)
  2. bmi = weight / ((height / 100) ** 2)
  3. Drop: height, weight, age
  4. X column order: [id, gender, ap_hi, ap_lo, cholesterol, gluc,
                      smoke, alco, active, age_years, bmi]

The saved pipeline (cardio_advanced_pipeline.pkl) is a full sklearn Pipeline:
  Pipeline([("scaler", StandardScaler()), ("model", HistGradientBoostingClassifier(...))])
It handles scaling internally — DO NOT scale before calling pipeline.predict().
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Load the pipeline once at startup ──────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cardio_advanced_pipeline.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"✅  Pipeline loaded from: {MODEL_PATH}")
except FileNotFoundError:
    pipeline = None
    print(f"⚠️   Pipeline NOT found at {MODEL_PATH}. Copy your .pkl file there.")

# ── Exact feature order as trained in week_2.ipynb ─────────────────────────────
# Cell 17: X = df_clean.drop(columns=['cardio'], axis=1)
# df_clean at that point: id, gender, ap_hi, ap_lo, cholesterol, gluc,
#                         smoke, alco, active, age_years, bmi
FEATURE_ORDER = [
    "id",          # kept in notebook (not dropped) — pass 0 for new predictions
    "gender",      # 1 = female, 2 = male
    "ap_hi",       # systolic blood pressure
    "ap_lo",       # diastolic blood pressure
    "cholesterol", # 1=normal, 2=above normal, 3=well above normal
    "gluc",        # 1=normal, 2=above normal, 3=well above normal
    "smoke",       # 0/1
    "alco",        # 0/1
    "active",      # 0/1
    "age_years",   # derived: int(age_days / 365)
    "bmi",         # derived: weight / (height_m ** 2)
]


def preprocess(data: dict) -> np.ndarray:
    """
    Replicate the EXACT feature engineering from week_2.ipynb.

    Accepts either:
      - age_days  (raw age in days, as in original dataset)
      - age_years (already converted) — preferred for UI input

    Accepts height (cm) and weight (kg) to derive BMI.
    """
    # ── Age ────────────────────────────────────────────────────────────────────
    if "age_days" in data:
        age_years = int(int(data["age_days"]) / 365)   # matches: .astype(int)
    else:
        age_years = int(data["age_years"])

    # ── BMI ────────────────────────────────────────────────────────────────────
    height_cm = float(data["height"])
    weight_kg = float(data["weight"])
    height_m  = height_cm / 100.0
    bmi       = weight_kg / (height_m ** 2)

    # ── Build row in EXACT column order ────────────────────────────────────────
    row = {
        "id":          0,                        # dummy — not a real feature
        "gender":      int(data["gender"]),       # must be int
        "ap_hi":       int(data["ap_hi"]),        # must be int
        "ap_lo":       int(data["ap_lo"]),        # must be int
        "cholesterol": int(data["cholesterol"]),  # must be int
        "gluc":        int(data["gluc"]),         # must be int
        "smoke":       int(data["smoke"]),        # must be int
        "alco":        int(data["alco"]),         # must be int
        "active":      int(data["active"]),       # must be int
        "age_years":   age_years,                 # int
        "bmi":         bmi,                       # float
    }

    # Build a single-row DataFrame with columns in training order
    df = pd.DataFrame([row], columns=FEATURE_ORDER)
    return df


@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "No JSON body received."}), 400

    # Validate required keys
    required = ["gender", "ap_hi", "ap_lo", "cholesterol",
                "gluc", "smoke", "alco", "active",
                "height", "weight"]
    age_provided = "age_days" in body or "age_years" in body
    missing = [k for k in required if k not in body]
    if not age_provided:
        missing.append("age_days OR age_years")
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        df = preprocess(body)

        # pipeline.predict() calls scaler.transform() then model.predict() internally
        prediction = int(pipeline.predict(df)[0])
        probability = float(pipeline.predict_proba(df)[0][1])  # P(cardio=1)

        return jsonify({
            "prediction": prediction,
            "probability": round(probability * 100, 2),   # as percentage
            "result": "High Risk" if prediction == 1 else "Low Risk",
            "features_used": df.to_dict(orient="records")[0],  # debug: echo back
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None,
        "feature_order": FEATURE_ORDER,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)