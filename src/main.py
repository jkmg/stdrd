from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from joblib import load
#from pycaret.regression import *

st.write(""" ### Determination of STD NMR Binding Epitopes using the Reduced Dataset Approach from the STD factors at 0.75 and 6 s """)

st.sidebar.header('Analytical Approach')

# Collects user input features into dataframe
uploaded_file1 = st.sidebar.file_uploader("Upload a CSV File", type=["csv"])
if uploaded_file1 is not None:
    df = pd.read_csv(uploaded_file1, header = None)
    df.columns=['Proton', 'STD_0p75s', 'STD_6s']
    df['ksat'] = (- np.log((df.STD_6s - df.STD_0p75s)/df.STD_6s))/0.75
    df['STD0'] = df.ksat*df.STD_6s
    df['Epitope'] = df.STD0/(np.max(df.STD0))*100
    df.drop(['STD_0p75s', 'STD_6s'], inplace=True, axis=1)

# Displays the user input features
st.subheader('Analytical Approach')

if uploaded_file1 is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded')

#st.sidebar.header('ML Approach')

# Collects user input features into dataframe
#uploaded_file2 = st.sidebar.file_uploader("Upload a CSV File (without column headers) containing 3 columns, the first with the proton names, then the experimental STD factors at 0.75s and finally the STD factros at 6s", type=["csv"])
#if uploaded_file2 is not None:
#    poly = joblib.load('/home/usuario/Documentos/Marie_Curie/Proyectos/STD_ML/Model_Deployment/polynomial_features_stdml_model')
#    model=joblib.load("/home/usuario/Documentos/Marie_Curie/Proyectos/STD_ML/Model_Deployment/extra_trees_best_sklearn_regressor")
#    df2 = pd.read_csv(uploaded_file2, header = None)
#    df2.rename(columns={0: 'Proton', 1: 'STD_0p75s', 2:'STD_6s'}, inplace=True)
#    df2["feature1"] = df2.STD_0p75s/df2.STD_6s
#    df2["feature2"] = np.exp(df2.STD_0p75s/df2.STD_6s)
#    dfval_poly = poly.transform(df2.loc[:,"STD_0p75s":"feature2"])
#    dfval_poly = pd.DataFrame(dfval_poly)
#    pred = model.predict(dfval_poly)
#    pred = pd.DataFrame(pred, columns=['STD0_ML'])
#    pred['Proton'] = df2.Proton.values
#    pred = pred.reindex(columns=['Proton', 'STD0_ML'])
#    pred['Epitope_ML'] = pred.STD0_ML/(np.max(pred.STD0_ML))*100

# Displays the user input features
#st.subheader('ML Approach')

#if uploaded_file2 is not None:
#    st.write(pred)
#else:
#    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
