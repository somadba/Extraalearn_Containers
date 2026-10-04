# Import necessary libraries
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
extraslearn_predictor_api = Flask("Extraalearn Customer Predictor")

# Load the trained machine learning model
model = joblib.load("extraalearn_edtech_model_v1_0.joblib")

# Define a route for the home page (GET request)
@extraslearn_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Extraalearn Customer Prediction API!"

# Define an endpoint for single Customer prediction (POST request)
@extraslearn_predictor_api.post('/v1/customer')
def predict_customer():
    """
    This function handles POST requests to the '/v1/customer' endpoint.
    It expects a JSON payload containing customer details and returns
    the predicted customer as a JSON response.
    """
    # Get the JSON data from the request body
    customer_data = request.get_json()

    # Extract relevant features from the JSON data
    features = {
        'age': customer_data['age'],
        'website_visits': customer_data['website_visits'],
        'time_spent_on_website': customer_data['time_spent_on_website'],
        'page_views_per_visit': customer_data['page_views_per_visit'],
        'current_occupation_Student': customer_data['current_occupation_Student'],
        'current_occupation_Unemployed': customer_data['current_occupation_Unemployed'],
        'first_interaction_Website': customer_data['first_interaction_Website'],
        'profile_completed_Low': customer_data['profile_completed_Low'],
        'profile_completed_Medium': customer_data['profile_completed_Medium'],
        'last_activity_Phone_Activity': customer_data['last_activity_Phone_Activity'],
        'last_activity_Website_Activity': customer_data['last_activity_Website_Activity'],
        'print_media_type1_Yes': customer_data['print_media_type1_Yes'],
        'print_media_type2_Yes': customer_data['print_media_type2_Yes'],
        'digital_media_Yes': customer_data['digital_media_Yes'],
        'educational_channels_Yes': customer_data['educational_channels_Yes'],
        'referral_Yes': customer_data['referral_Yes']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([features])

    # Make prediction as to the visitor turns to paid customer
    #prediction = model.predict(input_data)[0]

    input_data = pd.DataFrame([features])

    #raw = model.predict(input_data)[0]
    #prediction = raw.item() if hasattr(raw, "item") else raw
    # Return Prediction
    #return jsonify({'Visitor turned to PAID customer': prediction})


    input_data = pd.DataFrame([features])

    # Class prediction (0 or 1)
    raw_pred = model.predict(input_data)[0]
    prediction = raw_pred.item() if hasattr(raw_pred, "item") else raw_pred

    # Probability of class 1 (converts to paid customer)
    raw_proba = model.predict_proba(input_data)[0][1]
    confidence = raw_proba.item() if hasattr(raw_proba, "item") else raw_proba

    # Return prediction + confidence
    return jsonify({
        'Visitor turned to PAID customer': prediction,
        'Confidence': round(confidence, 3)
    })


    # Return Prediction
    #return jsonify({'Will the visitor become customer? ': prediction})


# List of features the model was trained on — same 16 keys as predict_customer
MODEL_FEATURES = [
    'age', 'website_visits', 'time_spent_on_website', 'page_views_per_visit',
    'current_occupation_Student', 'current_occupation_Unemployed',
    'first_interaction_Website',
    'profile_completed_Low', 'profile_completed_Medium',
    'last_activity_Phone_Activity', 'last_activity_Website_Activity',
    'print_media_type1_Yes', 'print_media_type2_Yes',
    'digital_media_Yes', 'educational_channels_Yes', 'referral_Yes'
]


@extraslearn_predictor_api.post('/v1/customerbatch')
def predict_extraalearn_batch():
    """
    Handles POST requests to '/v1/customerbatch'.
    Expects a multipart CSV upload under the field name 'file'.
    Returns a dict of {customer_id: prediction}.
    """
    # 1. Guard: file field must be present
    if 'file' not in request.files:
        return jsonify({'error': "No file uploaded under field name 'file'"}), 400

    file = request.files['file']

    # 2. Read CSV
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Could not parse CSV: {e}'}), 400

    # 3. Normalise the ID column — accept either 'ID' or 'id'
    id_col = 'ID' if 'ID' in data.columns else ('id' if 'id' in data.columns else None)
    if id_col is None:
        return jsonify({'error': "CSV must contain an 'ID' or 'id' column"}), 400

    # 4. Replay preprocessing (same order as training)
    batch = data.drop(columns=[id_col])
    if "status" in batch.columns:                 # drop label if present
        batch = batch.drop(columns=["status"])

    to_get_dummies_for = ['current_occupation', 'first_interaction', 'profile_completed',
                          'last_activity', 'print_media_type1', 'print_media_type2',
                          'digital_media', 'educational_channels', 'referral']
    for col in to_get_dummies_for:
        batch[col] = batch[col].astype(str).str.strip().str.replace(' ', '_', regex=False)

    input_data = pd.get_dummies(batch, columns=to_get_dummies_for, drop_first=True)

    # 5. Align columns with the trained model
    # Any missing dummy columns are added as 0; extras are dropped.
    input_data = input_data.reindex(columns=MODEL_FEATURES, fill_value=0)

    # 6. Predict
    predictions = model.predict(input_data).tolist()
    # Cast numpy types to native so jsonify works
    predictions = [int(p) if hasattr(p, 'item') else p for p in predictions]

    customer_ids = data[id_col].tolist()
    customer_ids = [int(c) if hasattr(c, 'item') else c for c in customer_ids]

    output_dict = dict(zip(customer_ids, predictions))

    return jsonify(output_dict)


# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    #extraslearn_predictor_api.run(debug=True)
     extraslearn_predictor_api.run(host="0.0.0.0", port=7860)
