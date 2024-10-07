import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib
from streamlit_folium import st_folium
from weather_parser import get_weather_info
import json
from data_preprocessing import preprocess_new_data

def load_model(path):
    return joblib.load(path)

def create_input_df(input_fields, model):
    data = pd.DataFrame(input_fields).T
    # Assign column names to the DataFrame
    data.columns = model['input_cols']
    # Convert numeric columns to float  
    numeric_cols = model['numeric_cols']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')        
    return data

def predict(input_fields, model):
    data = create_input_df(input_fields, model)
    data = preprocess_new_data(data, model['input_cols'], model['imputer'], model['scaler'], model['encoder'])
    pred_proba = model['model'].predict_proba(data)
    return np.argmax(pred_proba), round(np.max(pred_proba), 2)
    
# Read locations, their ids and coordinates from a csv file
locations_df = pd.read_csv('./data/australian_locations.csv')
# List of all possible wind directions
wind_directions = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", None
]

# Add a title
st.header('Predict next-day rain in Australia')
# Add a map centered on Australia
m = folium.Map(location=[-28, 133.86], zoom_start=4, min_zoom=4)

# Add marker for each location to the map 
for index, row in locations_df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], tooltip=row['location']).add_to(m) 

# Render the Folium map  
map_data = st_folium(m, width=700, height=500)

# Check if a marker was clicked
selected_location = None
if map_data and map_data['last_object_clicked']:
    # Extract the tooltip info for the selected marker
    selected_location = map_data['last_object_clicked_tooltip']

# Fetch the data and display the fields once the location has been selected
if selected_location:
    # Get id of the selected location 
    locations = list(locations_df['location'])
    location_id = locations_df[locations_df['location'] == selected_location]['id'].iloc[0] 
    # get weather info from http://www.bom.gov.au/
    json_data = get_weather_info(location_id) 
    st.subheader("Today's Weather Observations")
    location = st.selectbox('Location', locations, index=locations.index(selected_location))

    min_temp = st.number_input('Minimum Temperature, °C', min_value=-20.0, max_value=50.0, 
                               value=float(json_data.get('MinTemp')) if json_data.get('MinTemp') else None)
    max_temp = st.number_input('Maximum Temperature, °C', min_value=-20.0, max_value=50.0, 
                               value=float(json_data.get('MinTemp')) if json_data.get('MaxTemp') else None)
    rainfall = st.number_input('Rainfall, mm', min_value=0.0, max_value=500.0, 
                               value=float(json_data.get('Rainfall')) if json_data.get('Rainfall') else None)
    evaporation = st.number_input('Evaporation, mm', min_value=0.0, max_value=200.0, 
                                  value=float(json_data.get('Evaporation')) if json_data.get('Evaporation') else None)
    sunshine = st.number_input('Sunshine, hours', min_value=0.0, max_value=16.0, 
                               value=float(json_data.get('Sunshine')) if json_data.get('Sunshine') else None)
    wind_gust_dir_value = str(json_data.get('WindGustDir')) if json_data.get('WindGustDir') else None
    wind_gust_dir = st.selectbox('Strongest Wind Gust Direction', wind_directions, index=wind_directions.index(wind_gust_dir_value))
    wind_gust_speed = st.number_input('Strongest Wind Gust Speed, km/h', min_value=0.0, max_value=200.0,
                                      value=float(json_data.get('WindGustSpeed')) if json_data.get('WindGustSpeed') else None)
    rain_today = 'Yes' if rainfall not in (0, None) else 'No'
    
    # Create two columns for 9 am and 3 pm fields
    col1, col2 = st.columns(2)
    
    # 9 am fields in the left column
    with col1:
        st.header("9 AM Readings")
        temp_9am = st.number_input('Temperature, °C', key=6, min_value=-20.0, max_value=50.0, 
                                   value=float(json_data.get('Temp9am')) if json_data.get('Temp9am') else None)
        humidity_9am = st.number_input('Humidity', key=3, min_value=0.0, max_value=100.0, 
                                       value=float(json_data.get('Humidity9am')) if json_data.get('Humidity9am') else None)
        pressure_9am = st.number_input('Pressure, hPa', key=4, min_value=870.0, max_value=1050.0,
                                       value=float(json_data.get('Pressure9am')) if json_data.get('Pressure9am') else None)
        cloud_9am = st.number_input('Cloud Amount, 8th', key=5, min_value=0.0, max_value=9.0, 
                                    value=float(json_data.get('Cloud9am')) if json_data.get('Cloud9am') else None)
        wind_dir_9am_value = str(json_data.get('WindDir9am')) if json_data.get('WindDir9am') else None
        wind_dir_9am = st.selectbox('Wind Direction', wind_directions, key=1, index=wind_directions.index(wind_dir_9am_value))
        wind_speed_9am = st.number_input('Wind Speed, km/h', key=2, min_value=0.0, max_value=200.0,
                                         value=float(json_data.get('WindSpeed9am')) if json_data.get('WindSpeed9am') else None)
        
    # 3 pm fields in the right column
    with col2:
        st.header("3 PM Readings")
        temp_3pm = st.number_input('Temperature, °C', key=12, min_value=-20.0, max_value=50.0, 
                                   value=float(json_data.get('Temp3pm')) if json_data.get('Temp3pm') else None)
        humidity_3pm = st.number_input('Humidity', key=9, min_value=0.0, max_value=100.0, 
                                       value=float(json_data.get('Humidity3pm')) if json_data.get('Humidity3pm') else None)
        pressure_3pm = st.number_input('Pressure, hPa', key=10, min_value=870.0, max_value=1050.0,
                                       value=float(json_data.get('Pressure3pm')) if json_data.get('Pressure3pm') else None)
        cloud_3pm = st.number_input('Cloud Amount, 8th', key=11, min_value=0.0, max_value=9.0, 
                                    value=float(json_data.get('Cloud3pm')) if json_data.get('Cloud3pm') else None)
        wind_dir_3pm_value = str(json_data.get('WindDir3pm')) if json_data.get('WindDir3pm') else None
        wind_dir_3pm = st.selectbox('Wind Direction', wind_directions, key=7, index=wind_directions.index(wind_dir_3pm_value))
        wind_speed_3pm = st.number_input('Wind Speed, km/h', key=8, min_value=0.0, max_value=200.0, 
                                         value=float(json_data.get('WindSpeed3pm')) if json_data.get('WindSpeed3pm') else None)
    
    submit = st.button('Predict')
    
    input_fields = [location, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_dir, 
                    wind_gust_speed, wind_dir_9am, wind_dir_3pm, wind_speed_9am, wind_speed_3pm, 
                    humidity_9am, humidity_3pm, pressure_9am, pressure_3pm, cloud_9am, 
                    cloud_3pm, temp_9am, temp_3pm, rain_today]
    if submit:
        model = load_model('models/aussie_rain.joblib')
        res, prob = predict(input_fields, model)
        st.write(f"Will it rain tomorrow? {'Yes' if res == 1 else 'No'}")
        st.write(f"Probability: {prob*100}%")
