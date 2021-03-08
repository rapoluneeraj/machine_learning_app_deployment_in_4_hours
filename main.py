import streamlit as st
from sklearn.preprocessing import StandardScaler
import os
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from pickle import dump
from pickle import load


def load_sidebar():
    st.sidebar.subheader("Predicting output")
    st.sidebar.success("")
@st.cache
def predict(df):
    standardized_data = StandardScaler().fit_transform(df)
    df2 = pd.DataFrame(standardized_data)
    classifier = load(open('pickle/classifier.pkl', 'rb'))
    prediction = classifier.predict(df2)

    return prediction
def main():

    # sidebar
    load_sidebar()

    # Title/ text
    st.title('Predicting output')

    st.text('')
    a = [st.text_input('Enter your variable value')]
    b = [st.text_input('Enter your another variable value')]
    column_names=['col1','col2']
    df = pd.DataFrame(columns = column_names)
    df['col1']=a
    df['col2']=b

    click = st.button('SUBMIT')
    if click:
        load_sidebar()
        predict(df)
        if predict==0:
            st.write('output is zero')
        else:
            st.write('output is one')

if(__name__ == '__main__'):
    main()
