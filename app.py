"""
Flask Web App for Disease Prediction
CodeAlpha ML Internship - Task 4
Run: python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model & scaler
MODEL_PATH   = 'models/best_model.pkl'
SCALER_PATH  = 'models/scaler.pkl'
FEATURES_PATH = 'models/feature_names.pkl'

model         = joblib.load(MODEL_PATH)   if os.path.exists(MODEL_PATH)   else None
scaler        = joblib.load(SCALER_PATH)  if os.path.exists(SCALER_PATH)  else None
feature_names = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else []


@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [float(data.get(f, 0)) for f in feature_names]
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        result = {
            "prediction": int(prediction),
            "label": "Diabetes Detected 🔴" if prediction == 1 else "No Diabetes ✅",
            "probability": round(float(probability) * 100, 2),
            "risk_level": (
                "High Risk"   if probability >= 0.7 else
                "Medium Risk" if probability >= 0.4 else
                "Low Risk"
            )
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chatbot endpoint — proxies to Anthropic Claude API.
    The frontend sends: { message, history, system }
    """
    import urllib.request, json as _json
    try:
        body = request.get_json()
        history = body.get('history', [])
        system  = body.get('system', 'You are a helpful health assistant.')

        # Keep last 10 turns only
        if len(history) > 10:
            history = history[-10:]

        payload = _json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 300,
            "system": system,
            "messages": history
        }).encode('utf-8')

        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                # API key from environment variable — set ANTHROPIC_API_KEY in terminal
                'x-api-key': os.environ.get('ANTHROPIC_API_KEY', '')
            },
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=15) as res:
            data = _json.loads(res.read())
        reply = data['content'][0]['text']
        return jsonify({"reply": reply})

    except Exception as e:
        # Fallback smart replies if API key not set
        msg = body.get('message', '').lower() if 'body' in dir() else ''
        fallbacks = {
            'diabetes': "Diabetes is a condition where your blood sugar stays too high. Type 2 is the most common and is often linked to diet and lifestyle. The good news? It's very manageable! 🌟",
            'bmi': "BMI stands for Body Mass Index. It's calculated as weight(kg) ÷ height(m)². A normal BMI is 18.5–24.9. Higher BMI can increase diabetes risk, but it's not the only factor! 😊",
            'glucose': "Glucose is your blood sugar level. After meals it rises; insulin brings it back down. In diabetes, this balance gets disrupted. Normal fasting glucose is below 100 mg/dL.",
            'insulin': "Insulin is a hormone your pancreas makes to help sugar enter your cells for energy. In diabetes, either not enough insulin is made (Type 1) or it stops working well (Type 2).",
            'prevent': "Great question! To lower diabetes risk: eat more vegetables & whole grains, reduce sugary drinks, walk 30 minutes daily, maintain a healthy weight, and get regular blood tests. You've got this! 💪",
            'symptom': "Common diabetes symptoms include: frequent thirst, needing to urinate often, feeling tired, blurry vision, and slow-healing wounds. But many people have no symptoms at all — that's why testing matters!",
        }
        for key, resp in fallbacks.items():
            if key in msg:
                return jsonify({"reply": resp})
        return jsonify({"reply": "That's a great question! For the best answers, please set your ANTHROPIC_API_KEY environment variable to enable the full AI chatbot. Meanwhile, feel free to ask about diabetes, BMI, glucose, insulin, or prevention tips! 😊"})


@app.route('/model-info')
def model_info():
    model_type = type(model).__name__ if model else "Not Loaded"
    return jsonify({
        "model_type": model_type,
        "features": feature_names,
        "num_features": len(feature_names)
    })


if __name__ == '__main__':
    if not model:
        print("⚠️  Model not found! Run 'python train_model.py' first.")
    else:
        print("✅ Model loaded successfully!")
        print("🌐 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True)
