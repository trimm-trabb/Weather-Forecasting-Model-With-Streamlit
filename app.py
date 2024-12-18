import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium import Icon
from typing import Tuple
from streamlit_folium import st_folium
from WeatherScraper import WeatherScraper
from RainPredictorModel import RainPredictorModel
from Preprocessor import Preprocessor

# Class to handle UI Components
class RainPredictionApp:
    def __init__(self, locations_file, model_path):
        self.locations_df = pd.read_csv(locations_file)
        self.model_manager = RainPredictorModel(model_path)
        self.wind_directions = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", None
        ]
        self.selected_location = None

    def display_header(self):
        st.header('Predict next-day rain in Australia')
        st.text("This application uses machine learning model to predict whether it will rain tomorrow in a given Australian location.")
        st.text('Select a location on the map to start:')

    def display_map(self):
        """Display a Folium map and handle location selection."""
        m = folium.Map(location=[-28, 133.86], zoom_start=4, min_zoom=4)

        for _, row in self.locations_df.iterrows():
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                tooltip=row['location'],
                icon=Icon(icon="location-pin", color="blue")  
            ).add_to(m)

        # Render the map and trigger rerun if necessary
        map_data = st_folium(m, width=700, height=500)
        
        if map_data and map_data['last_object_clicked']:
            self.selected_location = map_data['last_object_clicked_tooltip']

    def get_weather_data(self):
        """Fetch weather data for the selected location using WeatherScraper."""
        if self.selected_location:
            location_id = self.locations_df[self.locations_df['location'] == self.selected_location]['id'].iloc[0]
            scraper = WeatherScraper(location_id)
            return scraper.get_weather_info()
        return None

    def render_input_fields(self, weather_data):
        """Display user input fields dynamically based on weather data."""
        st.subheader("Today's Weather Observations")
    
        # Location selection dropdown
        location = st.selectbox(
            'Location',
            self.locations_df['location'].tolist(),
            index=self.locations_df['location'].tolist().index(self.selected_location)
        )
    
        # General weather fields
        min_temp = st.number_input(
            'Minimum Temperature, °C', 
            min_value=-20.0, max_value=50.0, 
            value=float(weather_data.get('MinTemp', 0.0)) if weather_data.get('MinTemp') else None
        )
        max_temp = st.number_input(
            'Maximum Temperature, °C', 
            min_value=-20.0, max_value=50.0, 
            value=float(weather_data.get('MaxTemp', 0.0)) if weather_data.get('MaxTemp') else None
        )
        rainfall = st.number_input(
            'Rainfall, mm', 
            min_value=0.0, max_value=500.0, 
            value=float(weather_data.get('Rainfall', 0.0)) if weather_data.get('Rainfall') else None
        )
        evaporation = st.number_input(
            'Evaporation, mm', 
            min_value=0.0, max_value=200.0, 
            value=float(weather_data.get('Evaporation', 0.0)) if weather_data.get('Evaporation') else None
        )
        sunshine = st.number_input(
            'Sunshine, hours', 
            min_value=0.0, max_value=16.0, 
            value=float(weather_data.get('Sunshine', 0.0)) if weather_data.get('Sunshine') else None
        )
    
        # Strongest Wind Gust fields
        wind_gust_dir_value = str(weather_data.get('WindGustDir')) if weather_data.get('WindGustDir') else None
        wind_gust_dir = st.selectbox(
            'Strongest Wind Gust Direction', 
            self.wind_directions, 
            index=self.wind_directions.index(wind_gust_dir_value)
        )
        wind_gust_speed = st.number_input(
            'Strongest Wind Gust Speed, km/h', 
            min_value=0.0, max_value=200.0, 
            value=float(weather_data.get('WindGustSpeed', 0.0)) if weather_data.get('WindGustSpeed') else None
        )
    
        # Rain Today field
        rain_today = 'Yes' if rainfall not in (0, None) else 'No'        
        st.text(f"Rain Today: {rain_today}")
    
        # Columns for 9 AM and 3 PM Readings
        col1, col2 = st.columns(2)
    
        # 9 AM Readings
        with col1:
            st.header("9 AM Readings")
            temp_9am = st.number_input(
                'Temperature, °C', key="temp_9am", 
                min_value=-20.0, max_value=50.0, 
                value=float(weather_data.get('Temp9am', 0.0)) if weather_data.get('Temp9am') else None
            )
            humidity_9am = st.number_input(
                'Humidity, %', key="humidity_9am", 
                min_value=0.0, max_value=100.0, 
                value=float(weather_data.get('Humidity9am', 0.0)) if weather_data.get('Humidity9am') else None
            )
            pressure_9am = st.number_input(
                'Pressure, hPa', key="pressure_9am", 
                min_value=870.0, max_value=1050.0, 
                value=float(weather_data.get('Pressure9am', 0.0)) if weather_data.get('Pressure9am') else None
            )
            cloud_9am = st.number_input(
                'Cloud Amount, 8ths', key="cloud_9am", 
                min_value=0.0, max_value=8.0, 
                value=float(weather_data.get('Cloud9am', 0.0)) if weather_data.get('Cloud9am') else None
            )
            wind_dir_9am = st.selectbox(
                'Wind Direction', self.wind_directions, 
                key="wind_dir_9am", 
                index=self.wind_directions.index(weather_data.get('WindDir9am', self.wind_directions[0]))
            )
            wind_speed_9am = st.number_input(
                'Wind Speed, km/h', key="wind_speed_9am", 
                min_value=0.0, max_value=200.0, 
                value=float(weather_data.get('WindSpeed9am', 0.0)) if weather_data.get('WindSpeed9am') else None
            )
    
        # 3 PM Readings
        with col2:
            st.header("3 PM Readings")
            temp_3pm = st.number_input(
                'Temperature, °C', key="temp_3pm", 
                min_value=-20.0, max_value=50.0, 
                value=float(weather_data.get('Temp3pm', 0.0)) if weather_data.get('Temp3pm') else None
            )
            humidity_3pm = st.number_input(
                'Humidity, %', key="humidity_3pm", 
                min_value=0.0, max_value=100.0, 
                value=float(weather_data.get('Humidity3pm', 0.0)) if weather_data.get('Humidity3pm') else None
            )
            pressure_3pm = st.number_input(
                'Pressure, hPa', key="pressure_3pm", 
                min_value=870.0, max_value=1050.0, 
                value=float(weather_data.get('Pressure3pm', 0.0)) if weather_data.get('Pressure3pm') else None
            )
            cloud_3pm = st.number_input(
                'Cloud Amount, 8ths', key="cloud_3pm", 
                min_value=0.0, max_value=8.0, 
                value=float(weather_data.get('Cloud3pm', 0.0)) if weather_data.get('Cloud3pm') else None
            )
            wind_dir_3pm = st.selectbox(
                'Wind Direction', self.wind_directions, 
                key="wind_dir_3pm", 
                index=self.wind_directions.index(weather_data.get('WindDir3pm', self.wind_directions[0]))
            )
            wind_speed_3pm = st.number_input(
                'Wind Speed, km/h', key="wind_speed_3pm", 
                min_value=0.0, max_value=200.0, 
                value=float(weather_data.get('WindSpeed3pm', 0.0)) if weather_data.get('WindSpeed3pm') else None
            )
            
            return [location, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_dir, 
                        wind_gust_speed, wind_dir_9am, wind_dir_3pm, wind_speed_9am, wind_speed_3pm, 
                        humidity_9am, humidity_3pm, pressure_9am, pressure_3pm, cloud_9am, 
                        cloud_3pm, temp_9am, temp_3pm, rain_today]

    def make_prediction(self, input_fields):
        """Make predictions using the model and display the result."""
        prediction, probability = self.model_manager.predict(input_fields)
        st.subheader(f"Will it rain tomorrow? {'Yes' if prediction == 1 else 'No'}")
        st.subheader(f"Probability: {probability * 100}%")

    def run(self):
        """Main method to run the app."""
        self.display_header()
        self.display_map()

        if self.selected_location:
            weather_data = self.get_weather_data()
            input_fields = self.render_input_fields(weather_data)

            if st.button('Predict'):
                self.make_prediction(input_fields)


# Main function to launch the app
if __name__ == '__main__':
    APP = RainPredictionApp('./data/australian_locations.csv', 'models/aussie_rain_lgbm.joblib')
    APP.run()
