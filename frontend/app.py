import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
#BACKEND_URL = "http://backend:7860" #### it’s the two-container case. 127.0.0.1 and host.docker.internal are both wrong for this setup. 

# Added because Containers reach each other by container name on a shared Docker network
import os
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860")

# Set the title of the Streamlit app
st.title("ExtraaLearn Customer Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for customer features
age = st.number_input("Age", min_value=1, max_value=100, value=30)
website_visits = st.number_input("Number of Website Visits", min_value=0, step=1, value=0)
time_spent_on_website = st.number_input("Time Spent on Website", min_value=0, step=1, value=0)
page_views_per_visit = st.number_input("Page Views per Visit", min_value=0, step=1, value=0)
current_occupation_Student=st.selectbox("Current Occupation - A Student?", [True,False])
current_occupation_Unemployed=st.selectbox("Current Occupation - Unemployed?", [True,False])
first_interaction_Website=st.selectbox("first_interaction - Website?", [True,False])
profile_completed_Low=st.selectbox("Profile Completed - Low?", [True,False])
profile_completed_Medium=st.selectbox("Profile Completed - Medium?", [True,False])
last_activity_Phone_Activity=st.selectbox("last_activity - Phone Activity?", [True,False])
last_activity_Website_Activity=st.selectbox("last_activity - Website Activity?", [True,False])
print_media_type1_Yes=st.selectbox("print_media_type1 - Yes?", [True,False])
print_media_type2_Yes=st.selectbox("print_media_type2 - Yes?", [True,False])
digital_media_Yes=st.selectbox("digital_media - Yes?", [True,False])
educational_channels_Yes=st.selectbox("educational_channels - Yes?", [True,False])
referral_Yes=st.selectbox("referral - Yes?", [True,False])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
'age': age,
'website_visits': website_visits,
'time_spent_on_website': time_spent_on_website,
'page_views_per_visit': page_views_per_visit,
'current_occupation_Student': current_occupation_Student,
'current_occupation_Unemployed': current_occupation_Unemployed,
'first_interaction_Website': first_interaction_Website,
'profile_completed_Low': profile_completed_Low,
'profile_completed_Medium': profile_completed_Medium,
'last_activity_Phone_Activity': last_activity_Phone_Activity ,
'last_activity_Website_Activity': last_activity_Website_Activity,
'print_media_type1_Yes': print_media_type1_Yes,
'print_media_type2_Yes': print_media_type2_Yes,
'digital_media_Yes': digital_media_Yes,
'educational_channels_Yes': educational_channels_Yes,
'referral_Yes': referral_Yes
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/customer", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Visitor turned to PAID customer']
        st.success(f"Visitor turned to PAID customer: {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/customerbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
