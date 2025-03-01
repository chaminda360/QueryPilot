import os
import pandas as pd
import streamlit as st

def upload_and_save_file():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data_folder = "./data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        file_path = os.path.join(data_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def load_csv(file_path):
    try:
        panda_df = pd.read_csv(file_path).fillna(0)
        st.write("### Dataset Preview")
        st.write(panda_df.head())
        return panda_df
    except Exception as e:
        st.error(f"Error processing the CSV file: {e}")
        return None
