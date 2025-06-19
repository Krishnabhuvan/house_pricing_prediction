from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from custom_transformers import CombinedAttributesAdder   # ✅ Import here

app = Flask(__name__)

# Load model and pipeline
model = joblib.load("models/house_price_model.pkl")
pipeline = joblib.load("models/data_pipeline.pkl")

FEATURE_COLUMNS = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'ocean_proximity'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def show_predict_form():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        transformed = pipeline.transform(df)
        prediction = model.predict(transformed)[0]
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
