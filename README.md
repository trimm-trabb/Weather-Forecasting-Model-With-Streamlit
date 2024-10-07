# Australia Rain Prediction App 

The Australia Rain Prediction App is a web application built with Streamlit that allows users to input weather-related data and predict whether it will rain tomorrow in a given Australian location. The predictions are made using Random Forest machine learning model trained on historical weather data from various regions.

## Features
Map Integration: Users can interact with a Folium map to select a location.

Data Retrieval: Today's weather observations for the selected location are fetched from http://www.bom.gov.au/ 

Interactive Input Fields: Users can update weather parameters.

Rain Prediction: Based on the input data, the app provides a prediction on whether it will rain tomorrow.

## Technologies
Streamlit: Frontend for creating the interactive web app.

Python: Backend language for data processing and machine learning model handling.

Folium: Map integration for location selection.

Pandas: Data manipulation and handling.

scikit-learn: Machine learning model for predicting rain.

Joblib: Used for saving and loading the machine learning model. This ensures efficient storage and retrieval of the trained model, enabling quick predictions based on user input.

## Dataset
The app uses a historical weather dataset that contains daily weather observations from various locations across Australia (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

## Model
The machine learning model is trained using historical weather data. The app uses a pre-trained Random Forest Classifier to make predictions. 
