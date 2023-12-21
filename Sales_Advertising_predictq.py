#kaler biru ni dlm linux, base python and writing utk generate py file
import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Simple Sales Prediction App") # kena ada # utk first title
st.write("This app predicts the amount of **Sales**!")

st.sidebar.header('User Input Parameters') #sidebar utk user interact

def user_input_features():
    TV = st.sidebar.slider('TV', 0.7, 296.4, 147.04) # value max , min, default
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 23.26) #user define function utk user interaction
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114.0, 30.55)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0]) #data kena ikut susunan column features yang ditrain
    return features

#call function
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


modelGaussianIris = pickle.load(open("AdvertisingLRmodel.h5", "rb"))


prediction = modelGaussianIris.predict(df)
prediction_proba = modelGaussianIris.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
