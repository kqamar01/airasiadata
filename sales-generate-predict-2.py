import streamlit as st
import pandas as pd
import seaborn as sns
import pickle


st.write("# Sales Prediction App")
st.write("This app predicts sales based on advertising budget.")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV Budget', 0.7, 296.4, 147.04)
    Radio = st.sidebar.slider('Radio Budget', 0.0, 49.6, 23.26)
    Newspaper = st.sidebar.slider('Newspaper Budget', 0.3, 114.0, 30.55)
    data = {'TV Budget': TV,
            'Radio Budget': Radio,
            'Newspaper Budget': Newspaper,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


loaded_modellinear = pickle.load(open("modellradv2.h5", "rb"))
prediction=loaded_modellinear.predict(df)

st.subheader('Prediction')
st.write(prediction)

