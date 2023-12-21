import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.externals import joblib  # For older scikit-learn versions, or use 'from joblib import load'
# For newer versions of scikit-learn, you can use 'import joblib' without the 'externals' module.

# Function to download the model file from GitHub
@st.cache  # Cache the download to avoid re-downloading on every run
def download_model():
    model_url = 'https://github.com/kqamar01/airasiadata/blob/5de8204aefba01aacc321512012b79d29253194d/modellradv.h5'  # Replace with your GitHub URL
    response = requests.get(model_url)
    model = joblib.load(BytesIO(response.content))
    return model

# Load the model
model = download_model()

# Page title and description
st.title('Sales Prediction App')
st.write('This app predicts sales based on advertising budget.')

# Sidebar for user input using sliders
st.sidebar.header('Input Data')

def user_input_features():
    TV = st.sidebar.slider('TV Budget', 0.7, 296.4, 147.04) 
    Radio = st.sidebar.slider('Radio Budget', 0.0, 49.6, 23.26) 
    Newspaper = st.sidebar.slider('Newspaper Budget', 0.3, 114.0, 30.55)

# Predict function using the loaded model
def predict_sales(tv, radio, newspaper):
    input_data = pd.DataFrame({'TV': [TV], 'Radio': [Radio], 'Newspaper': [Newspaper]})
    prediction = model.predict(input_data)
    return prediction[0]

if st.sidebar.button('Predict'):
    predicted_sales = predict_sales(tv_budget, radio_budget, newspaper_budget)
    st.header('Predicted Sales')
    st.write(f'For the given budget: TV=${tv_budget}, Radio=${radio_budget}, Newspaper=${newspaper_budget}, the predicted sales is: ${predicted_sales:.2f}')
