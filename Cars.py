# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 01:08:16 2021

@author: HP
"""

import pandas as pd
import pickle
with open('model_pickle','rb') as file:
    mp = pickle.load(file)
pd.set_option('max_columns',None)

import streamlit as st
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv('data.csv')
df.keys()

# go to anaconda prompt and use --->>>> cd path to change base folder path
# streamlit run filename.py in anaconda

Col_name=df.columns
df_test=pd.DataFrame(columns=Col_name)
df_test.drop(columns=['Power.1','Ownership.1','Unnamed: 0'],inplace=True)
col=df_test.columns

for i in df_test.columns:
    df_test.loc[0,i]=0


location_list=df.Location.unique().tolist()
location_list.remove('Pune')
Fuel_list=df.Fuel.unique().tolist()
Fuel_list.remove('CNG')
Make_list=df.Make.unique().tolist()
Make_list.remove('Volvo')

df_test.loc[0,'Location']=st.selectbox('Location',location_list)
df_test.loc[0,'Fuel']=st.selectbox('Fuel', Fuel_list)
df_test.loc[0,'Make']=st.selectbox('Make', Make_list)
df_test.loc[0,'Km']=st.text_input('distance travelled in km',value=100)
df_test.loc[0,'Mileage']=st.text_input('Mileage in km/l',value=20)
df_test.loc[0,'Engine']=st.text_input('Engine size in cc',value=1000)
df_test.loc[0,'Power']=st.text_input('Power size in bhp',value=112)
df_test.loc[0,'Age']=st.text_input('Age',value=10)
df_test.loc[0,'Ownership']=st.selectbox('Prior number of owners', df.Ownership.unique())

for i in df_test.columns[10:]:
    df_test.loc[0,i]=0
df_test.loc[0,'Location_'+str(df_test['Location'][0])]=1
df_test.loc[0,'Make_'+str(df_test['Make'][0])]=1
df_test.loc[0,'Fuel_'+str(df_test['Fuel'][0])]=1
print(str(df_test.keys()))
#['Location_Pune','Fuel_CNG','Make_Volvo']
df_test.drop(columns={'Price','Location','Fuel','Make'},inplace=True)
df_test[list(df_test)].astype('float')
if mp.predict(df_test)<=0:
    Price=str(50000)
else:
    Price=str(int(mp.predict(df_test)*100000))
st.text("Price o vehicle is "+Price)


