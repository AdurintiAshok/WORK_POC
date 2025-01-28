import re
from langchain_groq import ChatGroq
import pandas as pd
import streamlit as st
from config import GROQ_API_KEY

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="deepseek-r1-distill-llama-70b",
    max_retries=2,
)

def load_csv_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            st.error("Please upload a CSV file.")
            return pd.DataFrame() 
    else:
        return pd.DataFrame()

def validate_csv_columns(user_data):
    required_columns = ["User Name", "Date", "Hours", "Task"]
    if not all(column in user_data.columns for column in required_columns):
        return False, f"CSV file must contain the following columns: {', '.join(required_columns)}"
    return True, "CSV file is valid."

def parse_date_column(user_data):
    try:
        user_data["Date"] = pd.to_datetime(user_data["Date"], infer_datetime_format=True).dt.strftime("%Y-%m-%d")
        return user_data
    except Exception as e:
        st.error(f"Error parsing 'Date' column: {e}")
        return user_data

def get_user_work_details(user_name, date_str, user_data):
    query = (
        f"Using this data: {user_data.to_dict('records')}, provide only the following details for {user_name} on {date_str}: "
        f"1. Total hours worked.\n"
        f"2. What {user_name} worked.\n"
        f"If no information is available for the user, respond with 'Not worked on anything today.dont provide extra details'"
    )
    result = llm.invoke(query)
    text = f"{result.content}"
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned_text

st.title("Work Summary App")
st.markdown("### Upload your timesheet CSV file to get work details")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

user_data = load_csv_data(uploaded_file)

if not user_data.empty:
    is_valid, validation_message = validate_csv_columns(user_data)
    if is_valid:
        user_data = parse_date_column(user_data)
        st.success("CSV file loaded successfully!")
    else:
        st.error(validation_message)
else:
    st.warning("Please upload a CSV file.")

st.markdown("### Query Work Details")

user_name = st.text_input("Enter the user name:")

date = st.date_input("Select the date:")

if st.button("Submit"):
    if user_name and date and not user_data.empty:
        is_valid, validation_message = validate_csv_columns(user_data)
        if is_valid:
            date_str = date.strftime("%Y-%m-%d")
            with st.spinner('Processing your request...'):
                result = get_user_work_details(user_name, date_str, user_data)
                st.success(result)
        else:
            st.error(validation_message)
    else:
        st.warning("Please provide both user name and date to proceed.")