import streamlit as st
import pandas as pd
import numpy as np
from sklearn.externals import joblib  # For older scikit-learn versions, or use 'from joblib import load'
# For newer versions of scikit-learn, you can use 'import joblib' without the 'externals' module.

# Load the trained model
model = joblib.load('modellradv.h5')  # Load your trained model here

# Page title and description
st.title('Sales Prediction App')
st.write('This app predicts sales based on advertising budget.')

# Sidebar for user input
st.sidebar.header('Input Data')

tv = st.sidebar.slider('TV Budget', min_value=0.0, max_value=1000.0, step=1.0)
radio = st.sidebar.slider('Radio Budget', min_value=0.0, max_value=1000.0, step=1.0)
newspaper = st.sidebar.slider('Newspaper Budget', min_value=0.0, max_value=1000.0, step=1.0)

# Predict function using the loaded model
def predict_sales(tv, radio, newspaper):
    # Create a DataFrame with the user input
    input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
    # Make the prediction
    prediction = model.predict(input_data)
    return prediction[0]  # Return the predicted sales value

if st.sidebar.button('Predict'):
    predicted_sales = predict_sales(tv, radio, newspaper)
    st.header('Predicted Sales')
    st.write(f'For the given budget: TV=${tv}, Radio=${radio}, Newspaper=${newspaper}, the predicted sales is: ${predicted_sales:.2f}')
