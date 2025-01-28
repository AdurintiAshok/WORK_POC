from uuid import uuid4
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
import pandas as pd
import streamlit as st
from config import GROQ_API_KEY
from io import StringIO
from datetime import datetime

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

def validate_csv_columns(user_data):
    required_columns = ["User Name", "Date"]
    if not all(column in user_data.columns for column in required_columns):
        return False, f"CSV file must contain the following columns: {', '.join(required_columns)}"
    return True, "CSV file is valid."

def parse_date_column(user_data):
    """
    Parse the 'Date' column into a standardized format (YYYY-MM-DD).
    """
    try:
        # Try to parse the 'Date' column into datetime, then format it as YYYY-MM-DD
        user_data["Date"] = pd.to_datetime(user_data["Date"], infer_datetime_format=True).dt.strftime("%Y-%m-%d")
        return user_data
    except Exception as e:
        st.error(f"Error parsing 'Date' column: {e}")
        return user_data

def filter_data_by_user_and_date(user_data, user_name, date_str):
    """
    Filter the data for the specific user and date.
    """
    filtered_data = user_data[(user_data["User Name"] == user_name) & (user_data["Date"] == date_str)]
    return filtered_data

def get_user_work_details(user_name, date_str, filtered_data):
    if filtered_data.empty:
        return "Not worked on anything today."

    total_hours = filtered_data["Hours"].sum()
    tasks = filtered_data["Task"].tolist()

    response = (
        f"Total hours worked on {date_str}: {total_hours}\n"
        f"What {user_name} worked on {date_str}: {', '.join(tasks)}"
    )
    return response

st.title("Work Summary App")
st.markdown("### Upload your timesheet CSV file to get work details")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

user_data = load_csv_data(uploaded_file)

if not user_data.empty:
    is_valid, validation_message = validate_csv_columns(user_data)
    if is_valid:
        # Parse the 'Date' column into a standardized format
        user_data = parse_date_column(user_data)
        st.success("CSV file loaded successfully!")
        st.write("Sample data:", user_data.head())  # Debugging: Display sample data
    else:
        st.error(validation_message)
else:
    st.warning("Please upload a CSV file.")

st.markdown("### Query Work Details")

user_name = st.text_input("Enter the user name:")

date = st.date_input("Select the date:")

if st.button("GetNow"):
    if user_name and date and not user_data.empty:
        is_valid, validation_message = validate_csv_columns(user_data)
        if is_valid:
            if user_name in user_data["User Name"].values:
                # Convert the user input date to the same format as the CSV
                date_str = date.strftime("%Y-%m-%d")
                # Filter the data for the specific user and date
                filtered_data = filter_data_by_user_and_date(user_data, user_name, date_str)
                with st.spinner('Processing your request...'):
                    result = get_user_work_details(user_name, date_str, filtered_data)
                    st.success(result)
            else:
                st.warning(f"User name '{user_name}' does not exist in the provided data.")
        else:
            st.error(validation_message)
    else:
        st.warning("Please provide both user name and date to proceed.")