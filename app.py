from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model, scaler, and imputation means
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('Scalar.joblib')
imputation_means = joblib.load('imputation_means.joblib')

# Define the expected feature order for consistent preprocessing
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert input data to a Pandas DataFrame
        # Ensure the order of columns matches the training data
        input_df = pd.DataFrame([data], columns=feature_cols)

        # Apply imputation for 0 values in specified columns, using the loaded means
        for col, mean_val in imputation_means.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].replace(0, mean_val)

        # Scale the input features using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)[:, 1]

        # Return the prediction as a JSON response
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_of_diabetes': float(prediction_proba[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
  print("Starting prediction API with preprocessing and model inference...") 
  app.run(debug=True)

print("Complete Flask application code with runner.")
