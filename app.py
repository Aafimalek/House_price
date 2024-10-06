import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Set up the Streamlit app layout
st.set_page_config(page_title="Bangalore Home Price Prediction", layout="wide")

# Title and Description
st.title("🏠 Bangalore Home Price Prediction")
st.markdown("""
This application predicts the home prices in Bangalore based on various factors such as:
- **Location**
- **Total Square Feet**
- **Number of Bathrooms**
- **BHK (Number of Bedrooms)**

Provide the house details on the left sidebar and click **Predict** to view the estimated price.
""")

# Sidebar for user inputs
st.sidebar.header("🏘️ House Details")
st.sidebar.write("Enter the details below to predict the home price:")

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('banglore_home_prices_model.pickle', 'rb') as file:
        return pickle.load(file)

# Load column data
@st.cache_resource
def load_columns():
    with open("columns.json", "r") as f:
        return json.load(f)['data_columns']

# Load model and columns
model = load_model()
data_columns = load_columns()
location_columns = data_columns[3:]  # Location columns start after the first 3 features

# Create input fields for user to fill in
location = st.sidebar.selectbox("📍 Select Location", sorted(location_columns))
total_sqft = st.sidebar.slider("🏠 Total Square Feet", min_value=300, max_value=10000, step=50, value=1000)
bathrooms = st.sidebar.number_input("🚽 Number of Bathrooms", min_value=1, max_value=10, step=1, value=2)
bhk = st.sidebar.number_input("🛏️ BHK (Number of Bedrooms)", min_value=1, max_value=10, step=1, value=2)

# Helper function to generate input array for the model
def create_input_array(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in location_columns:
        loc_index = data_columns.index(location.lower())
        x[loc_index] = 1
    return x

# Prediction and display result
if st.sidebar.button("🔍 Predict Price"):
    try:
        # Generate input vector for prediction
        input_vector = create_input_array(location, total_sqft, bathrooms, bhk)

        # Predict using the model
        predicted_price = model.predict([input_vector])[0]

        # Display the result with formatting
        st.success(f"💰 The estimated price for the house is: **₹ {predicted_price:.2f} lakhs**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Additional Information
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Additional Info")
st.sidebar.write("""
- Model used: **LinearRegression**
- Prediction Unit: **Price in Lakhs**
- Dataset: **Bangalore Home Prices**
""")
