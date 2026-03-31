from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model from the pickle file
model_path = os.path.join(os.path.dirname(__file__), 'model', 'upi_fraud_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return jsonify({"message": "UPI Fraud Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Extract features
        features = {
            'Transaction_Type': data.get('transactionType', ''),
            'Payment_Gateway': data.get('paymentGateway', ''),
            'Transaction_City': data.get('transactionCity', ''),
            'Transaction_State': data.get('transactionState', ''),
            'Merchant_Category': data.get('merchantCategory', ''),
            'amount': float(data.get('amount', 0)),
            'Year': int(data.get('year', 2023)),
            'Month': data.get('month', '')
        }
        
        # Process the input data (perform the same preprocessing as training data)
        # For simplicity, we'll create one-hot encoded features manually
        # In production, you should use the same encoder that was used during training
        
        # Example: Create a DataFrame with one row
        input_df = pd.DataFrame([features])
        
        # Preprocess the data (this is simplified for demonstration)
        # In a real application, you would apply the same transformations as during training
        # Convert categorical variables to one-hot encoding
        input_df_encoded = pd.get_dummies(input_df, columns=[
            'Transaction_Type', 
            'Payment_Gateway', 
            'Transaction_City', 
            'Transaction_State', 
            'Merchant_Category'
        ], drop_first=False)
        
        # Ensure all columns from training are present
        # This part should be replaced with proper handling of feature columns
        # For a real application, you should save the column names during training
        
        # Make prediction
        probability = model.predict_proba(input_df_encoded)[:, 1][0]
        prediction = 1 if probability > 0.5 else 0
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'result': 'Fraud' if prediction == 1 else 'Genuine',
            'confidence': f"{probability * 100:.2f}%" if prediction == 1 else f"{(1-probability) * 100:.2f}%"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# For demo purposes, we'll add a simpler endpoint that doesn't rely on model preprocessing
@app.route('/demo-predict', methods=['POST'])
def demo_predict():
    try:
        data = request.json
        
        # Use simple rules for demonstration purposes
        amount = float(data.get('amount', 0))
        transaction_type = data.get('transactionType', '')
        payment_gateway = data.get('paymentGateway', '')
        
        # Simple rule-based detection for demo
        is_fraud = (
            amount > 500 or 
            transaction_type in ['Investment', 'Refund'] or
            payment_gateway == 'CRED' and amount > 200
        )
        
        probability = 0.85 if is_fraud else 0.15
        
        result = {
            'prediction': 1 if is_fraud else 0,
            'probability': probability,
            'result': 'Fraud' if is_fraud else 'Genuine',
            'confidence': f"{probability * 100:.2f}%" if is_fraud else f"{(1-probability) * 100:.2f}%"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)