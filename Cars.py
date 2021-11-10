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

# go to anaconda prompt and use --->>>> cd path to change base folder path
# streamlit run filename.py in anaconda

Col_name=df.columns
df_test=pd.DataFrame(columns=Col_name)
df_test.drop(columns=['Power.1','Ownership.1','Unnamed: 0'],inplace=True)
col=df_test.columns

for i in df_test.columns:
    df_test.loc[0,i]=0

  
st.title('Used-Car Price Calculator')
st.sidebar.title('Input Feature')

Pt=pd.pivot_table(df,index="Make",columns=None,aggfunc={'Price':'mean'})
Pt['F']=Pt.index
t=sns.barplot(x=Pt.F,y='Price',data=Pt)
t.fig.set_figwidth(12)
t.fig.set_figheight(6)
plt.show()
st.pyplot()

location_list=df.Location.unique().tolist()
location_list.remove('Pune')
Fuel_list=df.Fuel.unique().tolist()
Fuel_list.remove('CNG')
Make_list=df.Make.unique().tolist()
Make_list.remove('Volvo')

df_test.loc[0,'Location']=st.sidebar.selectbox('Location',location_list)
df_test.loc[0,'Fuel']=st.sidebar.selectbox('Fuel', Fuel_list)
df_test.loc[0,'Make']=st.sidebar.selectbox('Make', Make_list)
df_test.loc[0,'Km']=st.sidebar.text_input('distance travelled in km',value=100)
df_test.loc[0,'Mileage']=st.sidebar.text_input('Mileage in km/l',value=20)
df_test.loc[0,'Engine']=st.sidebar.text_input('Engine size in cc',value=1000)
df_test.loc[0,'Power']=st.sidebar.text_input('Power size in bhp',value=112)
df_test.loc[0,'Age']=st.sidebar.text_input('Age',value=10)
df_test.loc[0,'Ownership']=st.sidebar.selectbox('Prior number of owners', df.Ownership.unique())

for i in df_test.columns[10:]:
    df_test.loc[0,i]=0
df_test.loc[0,'Location_'+str(df_test['Location'][0])]=1
df_test.loc[0,'Make_'+str(df_test['Make'][0])]=1
df_test.loc[0,'Fuel_'+str(df_test['Fuel'][0])]=1
print(str(df_test.keys()))
#['Location_Pune','Fuel_CNG','Make_Volvo']
df_test.drop(columns={'Price','Location','Fuel','Make'},inplace=True)
df_test[list(df_test)].astype('float')

def call(number):
    return ("{:,}".format(number))
if mp.predict(df_test)<=0:
    Price=str(50,000)
else:
    Price=str(call(int(mp.predict(df_test)*100000)))


      
st.header("Price of vehicle")
st.metric(label="", value="₹ "+str(Price), delta="")

#>>>>>>>>>>>>>>>>>>>>Graphs

sd = st.selectbox(
        "Select a Plot", #Drop Down Menu Name
        [
            "Location-Wise Price ", #First option in menu
            "Distribution according to distance run",   #Seconf option in menu
            "Distribution according to vehicle age"
        ]
    )

if sd == "Distribution according to distance run":
        #plt.figure(figsize=(12, 6))
        g=sns.displot(data=df,x=df.Km, color= 'purple')
        g.set(xlim=(0,200000))
        g.fig.set_figwidth(12)
        g.fig.set_figheight(6)
        plt.show()
        st.pyplot()
    
elif sd == "Location-Wise Price ":
        plt.figure(figsize=(12, 6))
        plt.ylim(0,60)
        sns.boxplot(x=df.Location,y=df.Price,data=df,width=.6)
        plt.show()
        st.pyplot()
        
elif sd == "Distribution according to vehicle age":
        plt.figure(figsize=(12, 6))
        k=sns.displot(x="Age",y=df.Price,data=df, hue=1,aspect=2.4)
        k.fig.set_figwidth(12)
        k.fig.set_figheight(5)
        plt.show()
        st.pyplot()



fig1 = sns.boxplot(x=df.Location,y=df.Price,data=df,width=.6)
plt.ylim(0,60) 



# location_list=df.Location.unique().tolist()
# location_list.remove('Pune')
# Fuel_list=df.Fuel.unique().tolist()
# Fuel_list.remove('CNG')
# Make_list=df.Make.unique().tolist()
# Make_list.remove('Volvo')

# df_test.loc[0,'Location']=st.sidebar.selectbox('Location',location_list)
# df_test.loc[0,'Fuel']=st.sidebar.selectbox('Fuel', Fuel_list)
# df_test.loc[0,'Make']=st.sidebar.selectbox('Make', Make_list)
# df_test.loc[0,'Km']=st.sidebar.text_input('distance travelled in km',value=100)
# df_test.loc[0,'Mileage']=st.sidebar.text_input('Mileage in km/l',value=20)
# df_test.loc[0,'Engine']=st.sidebar.text_input('Engine size in cc',value=1000)
# df_test.loc[0,'Power']=st.sidebar.text_input('Power size in bhp',value=112)
# df_test.loc[0,'Age']=st.sidebar.text_input('Age',value=10)
# df_test.loc[0,'Ownership']=st.sidebar.selectbox('Prior number of owners', df.Ownership.unique())

# for i in df_test.columns[10:]:
#     df_test.loc[0,i]=0
# df_test.loc[0,'Location_'+str(df_test['Location'][0])]=1
# df_test.loc[0,'Make_'+str(df_test['Make'][0])]=1
# df_test.loc[0,'Fuel_'+str(df_test['Fuel'][0])]=1
# print(str(df_test.keys()))
# #['Location_Pune','Fuel_CNG','Make_Volvo']
# df_test.drop(columns={'Price','Location','Fuel','Make'},inplace=True)
# df_test[list(df_test)].astype('float')

# def call(number):
#     return ("{:,}".format(number))
# if mp.predict(df_test)<=0:
#     Price=str(50,000)
# else:
#     Price=str(call(int(mp.predict(df_test)*100000)))


      
# st.header("Price of vehicle")
# st.metric(label="", value="₹ "+str(Price), delta="")
