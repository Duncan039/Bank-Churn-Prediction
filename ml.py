import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
st.set_page_config(layout="wide")
# App Layout
st.title('Bank Customer Churn Predictor')
st.image("BANK.jpg")
st.header('Enter the following customer Features:')

# load up the model
model = RandomForestClassifier()
model = joblib.load('mymodel')

# cache the model for easier loading
# function to take in features or variables
# standardise the inputs


# inputs
Geography = st.number_input('Geography,France=1,Germany=2,Spain=3', min_value=1, max_value=3, value=1)
Gender = st.number_input('Gender of the Customer,Male=1, Female=0', min_value=0, max_value=1, value=1)
IsActiveMember = st.number_input("Whether Customer is Active,Yes=1, No=0", min_value=0, max_value=1, value=1)
Age = st.number_input("Age of the Customer:", min_value=18, max_value=200, value=18)
Tenure = st.number_input("Enter Tenure:", min_value=0, max_value=1000, value=1)
EstimatedSalary = st.number_input("EstimatedSalary of the Customer:", min_value=500, max_value=1000000000, value=500)
CreditScore = st.number_input("CreditScore of the Customer:", min_value=0, max_value=1000000000, value=1)
# convert strings to numbers

prediction1 = pd.DataFrame([[Age, Tenure, CreditScore, IsActiveMember, EstimatedSalary, Geography, Gender]],
                           columns=["Age", "Tenure", "CreditScore", "IsActiveMember", "EstimatedSalary", "Geography",
                                    "Gender"])
# prediction2 = model.predict(pd.DataFrame([CreditScore, Geography, Gender, Age, Balance, IsActiveMember, EstimatedSalary]))
st.write("Sample DataFrame")
st.dataframe(prediction1)

scale = StandardScaler()
scaled = pd.DataFrame(scale.fit_transform(prediction1),
                      columns=["Age", "Tenure", "CreditScore", "IsActiveMember", "EstimatedSalary", "Geography",
                               "Gender"])
b = st.button("Predict Customer Churn")
if b:
    outcome = pd.DataFrame(scale.fit_transform(prediction1),
                       columns=["Age", "Tenure", "CreditScore", "IsActiveMember",
                                "EstimatedSalary",
                                "Geography",
                                "Gender"])
    st.write(f"Model Prediction is {model.predict(prediction1)}")
