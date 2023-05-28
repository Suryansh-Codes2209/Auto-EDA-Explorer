import time
from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("kisspng-deep-learning-machine-learning-artificial-neural-n-networking-5ace71b430f429.9366641915234789642005.png")
    st.title("Auto-EDA-Explorer")
    choice = st.radio("Nav-Panel", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")
    st.info("Made by Suryansh Chaudhary ")

if choice == "Upload":
    st.title("📚Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")

    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

        with st.spinner("Loading..."):
            time.sleep(5)
            st.success("Done!")

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(1):
            time.sleep(10)
            my_bar.progress(percent_complete + 50, text=progress_text)
        setup(df, target=chosen_target, remove_outliers=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        my_bar.progress(percent_complete + 100, text=progress_text)
        my_bar.empty()
        st.success('Model Generated ⚡', icon="✅")
        st.snow()


if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
