from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
import pandas as pd
import streamlit as st
from config import GROQ_API_KEY
from io import StringIO

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768",
    max_retries=2,
)

def load_csv_data(uploaded_file):
    if uploaded_file is not None:
         return pd.read_csv(uploaded_file)
    else:
        return pd.DataFrame()

def get_user_work_details(user_name, date, user_data):
    query = (
        f"Using this data: {user_data}, provide only the following details for {user_name} on {date}: "
        f"1. Total hours worked. "
        f"2. What She/He worked on.If no information is available, respond with 'Not worked on anything today."
        f"Do not include any additional explanations or details."
    )
    result = llm.invoke(query)
    return result.content

st.title("Work Summary App")
st.markdown("### Upload your timesheet CSV file to get work details")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


user_data = load_csv_data(uploaded_file)

if not user_data.empty:
    st.success("CSV file loaded successfully!")
else:
    st.warning("Please upload a CSV file.")

st.markdown("### Query Work Details")


user_name = st.text_input("Enter the user name:")


date = st.date_input("Select the date:")


if st.button("GetNow"):
    if user_name and date and not user_data.empty:

        if user_name in user_data["User Name"].values:
            with st.spinner('Processing your request...'):
                result = get_user_work_details(user_name, date, user_data)
                st.success(result)
        else:
            st.warning(f"User name '{user_name}' is not existed  in the provided data.")
    else:
        st.warning("Please provide both user name and date to proceed.")