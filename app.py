import streamlit as st
import pickle
import numpy as np
import json

# Load model and data
def load_model():
    with open("artifacts/banglore_home_price_prediction_model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/columns.json", "r") as f:
        data_columns = json.load(f)["data_columns"]
        locations = data_columns[3:]
    return model, locations, data_columns

model, locations, data_columns = load_model()

# Function to predict price
def get_estimated_price(location, sqft, bhk, bath):
    loc_index = data_columns.index(location.lower()) if location.lower() in data_columns else -1
    x = np.zeros(len(data_columns))
    x[0], x[1], x[2] = sqft, bath, bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)

# Streamlit UI
st.set_page_config(page_title="Bangalore Home Price Prediction", layout="centered")

st.title("🏠 Bangalore Home Price Prediction")

# CSS Styling
st.markdown(
    """
    <style>
        .stTextInput, .stNumberInput, .stSelectbox, .stButton button {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
        }
        .css-1d391kg { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Input Fields
sqft = st.text_input("Area (Square Feet)", "", key="sqft", placeholder="Enter area in sqft", help="Enter the total square feet")
bhk = st.radio("BHK", [1, 2, 3, 4, 5], index=1, horizontal=True)
bath = st.radio("Bathrooms", [1, 2, 3, 4, 5], index=1, horizontal=True)
location = st.selectbox("Select Location", locations)

# Predict Button
if st.button("Estimate Price 💰"):
    price = get_estimated_price(location, float(sqft), bhk, bath)
    st.success(f"Estimated Price: ₹ {price} Lakh")

