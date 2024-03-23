import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import joblib 
import numpy as np
#import plotly.express as px
import sklearn
#import seaborn as sns

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
model= joblib.load('Stroke_models.pkl')


st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: helvetica'>STROKE PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By UGOCHUKWU OBINNA AUGUSTINE</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('stroke.jpg')
st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>PROJECT OVERVIEW</h4>", unsafe_allow_html = True)
st.write('The goal of this project is to develop a predicitive model that predicts stroke in patients using certain parameters. By levering machine learning techniques, we aim to provide insights into the factors that exposes patients to Stroke, empowering health practicioners to make informed decision')

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data, use_container_width=True)

st.sidebar.image('the stroke.jpg' ,caption ='Welcome Patient')

gen =     st.sidebar.number_input('gender')
ag = st.sidebar.number_input('age ')
hyper = st.sidebar.number_input('hypertension')
heart =  st.sidebar.number_input('heart_disease')
evmar =  st.sidebar.number_input('ever_married')
wort =    st.sidebar.number_input('work_type')
res_t =    st.sidebar.number_input('Residence_type')
av_gl =  st.sidebar.number_input('avg_glucose_level')
b_m =  st.sidebar.number_input('bmi')
sm_st =  st.sidebar.number_input('smoking_status')


st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>inputs VARIABLES</h4>", unsafe_allow_html = True)


inputs = pd.DataFrame()
inputs['gender'] = [gen]
inputs['age'] = [ag]
inputs['hypertension'] = [hyper]
inputs['heart_disease'] = [heart]
inputs['ever_married'] = [evmar]
inputs['work_type'] = [wort]
inputs['Residence_type'] = [res_t]
inputs['avg_glucose_level'] = [av_gl]
inputs['bmi'] = [b_m]
inputs['smoking_status'] = [sm_st]



st.dataframe(inputs,use_container_width=True)

# Model Prediction
pusher= st.button('Predict Stroke')
if pusher:
    predicted = model.predict(inputs) 
    if predicted[0]== 1:
        st.error(f'see your doctor') 
    else:
        st.success(f'Congrats, you are free')
