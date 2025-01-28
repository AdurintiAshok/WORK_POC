from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
import pandas as pd
import streamlit as st
from config import GROQ_API_KEY
from io import StringIO

llm = ChatGroq(
    temperature=0.6,
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768",
    max_retries=2,
)

def load_csv_data(uploaded_file):
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return pd.read_csv(stringio)
    else:
        return pd.DataFrame()

def get_user_work_details(user_name, date, user_data):
    query = (
    f"Given the provided data: {user_data}, provide the total hours {user_name} worked on {date} and a description of what {user_name} worked on. "
    f"If no data is available for the specified user, respond only with 'No data available for the given user or on given date.' Do not explain anything further."
)


    result = llm.invoke(query)
    return result.content

st.title("Work Summary App")
st.markdown("### Upload your timesheet CSV file to get work details")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

user_data = load_csv_data(uploaded_file)

if not user_data.empty:
    st.success("CSV file loaded successfully!")
    # st.dataframe(user_data.head()) 
else:
    st.warning("Please upload a CSV file.")

st.markdown("### Query Work Details")

if not user_data.empty:
    # Ensure column name matches exactly, including spaces
    column_name = "User Name" if "User Name" in user_data.columns else "UserName" if "UserName" in user_data.columns else None

    if column_name:
        usernames = user_data[column_name].unique().tolist()
    else:
        usernames = []
        st.warning("The uploaded file does not have a 'User Name' or 'UserName' column.")
else:
    usernames = []

user_name = st.selectbox("Select the user name:", options=usernames, key="user_name_select")

date = st.date_input("Select the date:")

if st.button("GetNow"):
    if user_name and date and not user_data.empty:
        with st.spinner('Processing your request...'):
            result = get_user_work_details(user_name, date, user_data)
            st.success(result)
    else:
        st.warning("Please provide both user name and date to proceed.")
