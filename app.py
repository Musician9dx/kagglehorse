import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf


df=pd.read_csv("D:/INeuron/Horse/kagglehorse/resource/data/train.csv")

file=open("D:/INeuron/Horse/kagglehorse/resource/model/transformer.pkl","rb")
transform=pickle.load(file)

st.title("Horse Health Prediction")
st.markdown("###### A Musician 9DX Project")

with st.container():
    st.selectbox(label="Surgery",options=["yes","no"])
    st.selectbox(label="Age",options=["Adult","Young"])
    st.selectbox(label="temperature of extremeties",options=["cool","warm","normal","None","cold"])
    st.selectbox(label="peripheral pulse",options=["increased","reduced","normal","None","absent"])
    st.selectbox(label="abdomo_appearance",options=['depressed' ,'mild_pain', 'extreme_pain', 'alert', 'severe_pain', "slight",'None'])
    st.selectbox(label="pain",options=['serosanguious', 'cloudy', 'clear', 'None'])
    st.selectbox(label="capillary_refill_time",options=['more_3_sec' ,'less_3_sec' ,'3','None' ,])
    st.selectbox(label="mucous_membrane",options=['dark_cyanotic' ,'pale_cyanotic' ,'pale_pink', 'normal_pink' ,'bright_pink','bright_red', 'None'])
    st.selectbox(label="abdomen",options=['distend_small', 'distend_large', 'normal' ,'firm' ,'None' ,'other'])
    st.selectbox(label="rectal_exam_feces",options= ['decreased', 'absent', 'normal', 'increased', 'serosanguious','None' ])
    st.selectbox(label="asogastric_reflux",options= ['less_1_liter', 'more_1_liter' ,'slight', 'None','none',])
    st.selectbox(label="nasogastric_tube",options= ['slight', 'significant', 'None', 'none'])
    st.selectbox(label="abdominal_distention",options= ['slight' ,'moderate' , 'severe', 'None','none'])
    st.selectbox(label="peristalsis",options= ['absent' ,'hypomotile' ,'normal' ,'hypermotile',  'distend_small',"None"])





st.title("Training Metrics")
st.dataframe(df)