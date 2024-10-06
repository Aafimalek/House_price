import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load the pre-trained model
with open('banglore_home_prices_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Load the columns.json file to get feature information
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']
    location_columns = data_columns[3:]  # Location columns start after the first 3 features

# Streamlit App Title
st.title("Bangalore Home Price Prediction")
st.write("This app predicts the home prices in Bangalore based on various factors like location, square footage, and more.")

# Sidebar for user inputs
st.sidebar.header("Provide the House Details:")
location = st.sidebar.selectbox("Select Location", location_columns)
total_sqft = st.sidebar.number_input("Total Square Feet", min_value=300, max_value=10000, step=10)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.sidebar.number_input("BHK (Number of Bedrooms)", min_value=1, max_value=10, step=1)

# Prediction button
if st.sidebar.button("Predict"):
    try:
        # Create an empty array of zeros with the length of the feature columns
        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bathrooms
        x[2] = bhk

        # Set the location index in the input vector to 1 if the location exists
        if location in location_columns:
            loc_index = data_columns.index(location.lower())
            x[loc_index] = 1

        # Make prediction using the model
        predicted_price = model.predict([x])[0]
        
        # Display the prediction result
        st.subheader(f"Predicted Price of the House: ₹{predicted_price:.2f} lakhs")
    except Exception as e:
        st.error(f"Error: {e}")

# Adding instructions at the bottom
st.write("Fill in the details in the sidebar and click **Predict** to see the house price.")
