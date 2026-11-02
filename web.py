
import streamlit as st
import pandas as pd
import pickle

# Load saved files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scalar.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Purchase Prediction")

st.title("🛒 Purchase Prediction")
st.write("Predict whether a user will purchase a product.")

# User inputs
gender = st.selectbox("Select Gender", ["Male", "Female"])

age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=25
)

salary = st.number_input(
    "Enter Estimated Salary",
    min_value=0,
    value=30000
)

# Prediction button
if st.button("Predict"):

    # Create dataframe
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'EstimatedSalary': [salary]
    })

    # One hot encoding
    input_data = pd.get_dummies(
        input_data,
        columns=['Gender'],
        dtype=int
    )

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scaling
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.success("User is likely to purchase the product! 🎉")
    else:
        st.warning("User is unlikely to purchase the product.")




