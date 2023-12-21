#kaler biru ni dlm linux, base python and writing utk generate py file
import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

# Page title and description
st.write("Sales Prediction App") 
st.write("This app predicts sales based on advertising budget.")

# Sidebar for user input
st.sidebar.header('User Input Parameters') 

def user_input_features():
    TV = st.sidebar.slider('TV Budget', 0.7, 296.4, 147.04) 
    Radio = st.sidebar.slider('Radio Budget', 0.0, 49.6, 23.26) 
    Newspaper = st.sidebar.slider('Newspaper Budget', 0.3, 114.0, 30.55)
    data = {'TV Budget': TV,
            'Radio Budget': Radio,
            'Newspaper Budget': Newspaper}
    features = pd.DataFrame(data, index=[0]) 
    return features


df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

modellradv = pickle.load(open("modellradv.h5", "rb"))

prediction = modellradv.predict(df)

st.subheader('Prediction')
st.write(prediction)
