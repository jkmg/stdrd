import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import *

st.write(""" # ML-based Prediction of STD NMR Binding Epitope from STD factors at 0.75 and 6 s """)

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload a CSV File (without column headers) containing 3 columns, the first with the proton names, then the experimental STD factors at 0.75s and finally the STD factros at 6s", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header = None)
    model=load_model("/home/usuario/Documentos/Marie_Curie/Proyectos/STD_ML/Model_Deployment/Extra_tree_regressor_pycaret_STD_0p75s_6s_WithNoise0p05_MAE_0p17")
    df.rename(columns={0: 'Proton', 1: 'STD_0p75s', 2:'STD_6s'}, inplace=True)
    df["feature1"] = df.STD_0p75s/df.STD_6s
    df["feature2"] = np.exp(df.STD_0p75s/df.STD_6s)
    df.index= df.Proton.values
    df.drop(["Proton"],inplace=True, axis=1)
    pred = predict_model(model, data=df)
    pred.rename(columns={'prediction_label':'STD0_ML'}, inplace=True)
    pred['Epitope_ML'] = pred.STD0_ML/(np.max(pred.STD0_ML))*100
    pred.drop(['STD_0p75s', 'STD_6s', 'feature1','feature2', 'STD0_ML'], inplace=True, axis=1)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(pred)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    #st.write(df)
