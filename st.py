import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Load the trained Random Forest model and scaler
rf_model = joblib.load('randomforest.joblib')
scaler = joblib.load('scaler.joblib')  # Load the saved scaler for consistent scaling

# Streamlit app header
st.title('Bike Demand Prediction')

# Create user input form with a date picker and time input
st.header('Enter the weather data and select a date/time to predict bike demand')

# Date Picker for selecting the date
selected_date = st.date_input('Select Date', datetime.today())

# Time input for the hour of the day (0-23)
selected_hour = st.number_input('Hour (0-23)', min_value=0, max_value=23)

# Collecting weather data
temp = st.number_input('Temperature (°C)', min_value=-50.0, max_value=50.0, format="%.2f")
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, format="%.2f")
solarradiation = st.number_input('Solar Radiation (W/m²)', min_value=0.0, max_value=2000.0, format="%.2f")
dew = st.number_input('Dew Point (°C)', min_value=-50.0, max_value=50.0, format="%.2f")
windspeed = st.number_input('Windspeed (km/h)', min_value=0.0, max_value=150.0, format="%.2f")
precip = st.number_input('Precipitation (mm)', min_value=0.0, max_value=50.0, format="%.2f")

# Create a "Predict" button
predict_button = st.button('Predict Demand')

if predict_button:
    # Prepare the input data
    input_data = pd.DataFrame([[selected_hour, temp, humidity, solarradiation, dew, windspeed, precip]],
                              columns=['hour', 'temp', 'humidity', 'solarradiation', 'dew', 'windspeed', 'precip'])

    # Scale the input data using the saved scaler
    input_scaled = scaler.transform(input_data)  # Use the loaded scaler for consistent scaling

    # Make prediction
    predicted_demand = rf_model.predict(input_scaled)

    
    # Display datetime for the prediction
    prediction_datetime = datetime.combine(selected_date, datetime.min.time()) + pd.Timedelta(hours=selected_hour)
    st.write(f'Prediction Time: {prediction_datetime.strftime("%Y-%m-%d %H:%M:%S")}')


    # Display the result
    st.write(f'Predicted Bike Demand: {predicted_demand[0]:.2f}')
    
 
