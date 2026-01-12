import streamlit as st
import pickle
import numpy as np

# Load model
with open("ev_type_model.pkl", "rb") as f:
    rf, le_make, le_target = pickle.load(f)

st.title("🚘 EV Type Classifier")

# Inputs
year = st.number_input("Model Year", 2010, 2025, 2022)
make = st.selectbox("Make", le_make.classes_)
msrp = st.number_input("Base MSRP", 10000, 200000, 40000)

# Encode inputs
make_enc = le_make.transform([make])[0]

# Predict
if st.button("Predict EV Type"):
    input_data = np.array([[year, make_enc, msrp]])
    prediction = rf.predict(input_data)
    ev_type = le_target.inverse_transform(prediction)

    st.success(f"Predicted EV Type: **{ev_type[0]}**")