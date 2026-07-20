import streamlit as st
import pandas as pd
import joblib


# Load saved model files
model = joblib.load("LR_ford_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")


# Page Configuration
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="wide"
)


# Sidebar
st.sidebar.title("Ford Car Price Prediction")
st.sidebar.write("Mini Project")
st.sidebar.write("Linear Regression Model")
st.sidebar.write("Developed using Streamlit")


# Main Title
st.title("Ford Car Price Prediction System")
st.write("Predict the selling price of a Ford car")

st.divider()


# Input Columns
col1, col2 = st.columns(2)


with col1:
    year = st.number_input(
        "Manufacturing Year",
        min_value=1996,
        max_value=2025,
        value=2018
    )

    mileage = st.number_input(
        "Mileage",
        min_value=0,
        max_value=300000,
        value=30000
    )

    tax = st.number_input(
        "Road Tax",
        min_value=0,
        max_value=600,
        value=150
    )


with col2:
    mpg = st.number_input(
        "MPG",
        min_value=0.0,
        max_value=150.0,
        value=55.40
    )

    engineSize = st.number_input(
        "Engine Size",
        min_value=0.0,
        max_value=6.0,
        value=1.50
    )


# Categorical Inputs
with col1:
    car_model = st.text_input(
        "Car Model",
        "Fiesta"
    )


with col2:
    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic",
            "Semi-Auto"
        ]
    )


fuelType = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid",
        "Electric"
    ]
)


# Prediction Button
predict = st.button("Predict Price")


# Prediction
if predict:

    input_df = pd.DataFrame({
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })


    # One Hot Encoding
    input_df = pd.get_dummies(input_df)


    # Match training columns
    input_df = input_df.reindex(
        columns=encoded_columns,
        fill_value=0
    )


    # Scale Numerical Columns
    numerical_cols = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]


    input_df[numerical_cols] = scaler.transform(
        input_df[numerical_cols]
    )


    # Prediction
    prediction = model.predict(input_df)


    # Display Result
    st.success(
        f"Predicted Price : £{prediction[0]:,.2f}"
    )


# Footer
st.divider()

st.caption(
    "Developed by Harish Marne | AIML Diploma Mini Project"
)