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
    prediction = model.predict(input_data)[0]

    # Return Prediction
    return jsonify({'Will the visitor become customer? ': prediction})


# Define an endpoint for batch prediction (POST request)
@extraslearn_predictor_api.post('/v1/customerbatch')
def predict_extraalearn_batch():
    """
    This function handles POST requests to the '/v1/customerbatch' endpoint.
    It expects a CSV file containing customer details for multiple properties
    and returns the predicted customer as a dictionary in the JSON response.
    """
    # Get the cleaned uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    data = pd.read_csv(file)

    # 2. Replay the *same* preprocessing, in the same order
    batch = data.drop(columns=["ID"])
    if "status" in batch.columns:              # in case the batch has labels
        batch = batch.drop(columns=["status"])

    to_get_dummies_for = ['current_occupation', 'first_interaction', 'profile_completed',
                          'last_activity', 'print_media_type1', 'print_media_type2',
                          'digital_media', 'educational_channels', 'referral']
    for col in to_get_dummies_for:
        batch[col] = batch[col].astype(str).str.strip().str.replace(' ', '_', regex=False)

    input_data = pd.get_dummies(batch, columns=to_get_dummies_for, drop_first=True)

    # Make predictions for all properties in the DataFrame (paid customer = True/False)
    prediction = model.predict(input_data).tolist()

    # Create a dictionary of predictions with customer IDs as keys
    customer_ids = data['id'].tolist()  # 'id' is the customer ID column
    output_dict = dict(zip(customer_ids, prediction))

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    #extraslearn_predictor_api.run(debug=True)
     extraslearn_predictor_api.run(host="0.0.0.0", port=7860)
