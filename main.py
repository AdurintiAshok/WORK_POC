from langchain_groq import ChatGroq
import pandas as pd
import streamlit as st
from config import GROQ_API_KEY

llm = ChatGroq(
    temperature=0.6,
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

def filter_data_by_user_and_date(user_data, user_name, date_str):

    user_data["User Name (Normalized)"] = user_data["User Name"].str.lower()
    user_name_normalized = user_name.lower()
    filtered_data = user_data[(user_data["User Name (Normalized)"] == user_name_normalized) & (user_data["Date"] == date_str)]
    return filtered_data.drop(columns=["User Name (Normalized)"])

def get_user_work_details(user_name, date_str, filtered_data):
    if filtered_data.empty:
        return "Not worked on anything today."
    query = (
        f"Using this data: {filtered_data.to_dict('records')}, provide only the following details for {user_name} on {date_str}: "
        f"1. Total hours worked on {date_str}. "
        f"2. What {user_name} worked on {date_str}. "
        f"If no information is available, respond with 'Not worked on anything today.' "
        f"Do not include any additional explanations or details."
    )
    result = llm.invoke(query)
    return result.content

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
            user_data["User Name (Normalized)"] = user_data["User Name"].str.lower()
            user_name_normalized = user_name.lower()
            if user_name_normalized in user_data["User Name (Normalized)"].values:
                date_str = date.strftime("%Y-%m-%d")
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