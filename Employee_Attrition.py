# Show in streamlit
import streamlit as st
import joblib
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

#two tabs
st.set_page_config(page_title="Employee Predictions", layout="wide")
st.title("Employee Predictions")


#tabs
tab1, tab2 = st.tabs(["Attrition Rate", "Performance Rating"])
with tab1:
    # Load the model
    model = joblib.load('employee_attrition_model.pkl')



    with st.form(key='input_form'):
        # Sample data for prediction
        sample_data = {
            'Age': [30],
            'Department': [3],
            'DistanceFromHome': [1],
            'EducationField': [2],
            'Gender': [0],
            'JobRole': [3],
            'MaritalStatus': [0],
            'MonthlyRate': [0.6780],
            'NumCompaniesWorked': [1],
            'OverTime': [1],
            'PerformanceRating': [2]
        }
        # Convert sample data to DataFrame
        sample_df = pd.DataFrame(sample_data)
        sample_data.keys()
        # Display the sample data
        st.write("Sample Data for Prediction:")
        st.write(sample_df)

        # Predict the attrition probability
        attrition_probability = model.predict_proba(sample_df)[:, 1][0]
        # Display the prediction result
        st.write(f"Predicted Attrition Probability: {attrition_probability:.2f}")


        # Create new data for prediction
        # Create the input form
        st.title("Employee Attrition Prediction")
        # st.write("Please fill in the details below:")
        # Create a form to collect user input
        st.subheader("Input Form")
        st.write("Please fill in the details below:")
        # Create a form to collect user input
        Age = st.number_input('Age', min_value=18, max_value=100, value=30)
        Department = st.selectbox('Department', options=['Sales', 'Research & Development', 'Human Resources'])
        DistanceFromHome = st.number_input('DistanceFromHome', min_value=0, max_value=100, value=5)
        EducationField = st.selectbox('EducationField', options=['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'])     
        Gender = st.selectbox('Gender', options=['Male','Female']),
        JobRole = st.selectbox('JobRole', options=['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Healthcare Representative']),
        MaritalStatus = st.selectbox('MaritalStatus', options=['Single', 'Married', 'Divorced']),
        MonthlyRate = st.number_input('MonthlyRate', min_value=0.0, max_value=10000.0, value=5000.0),
        NumCompaniesWorked = st.number_input('NumCompaniesWorked', min_value=0, max_value=10, value=2),
        OverTime = st.selectbox('OverTime', options=['Yes', 'No']),
        PerformanceRating = st.number_input('PerformanceRating', min_value=1, max_value=5, value=3)
        # Create a form to collect user input
        st.subheader("Input Form") 

        input_data = {
            'Age': Age,
            'Department': Department,
            'DistanceFromHome':DistanceFromHome,
            'EducationField': EducationField,
            'Gender': Gender,
            'JobRole': JobRole,
            'MaritalStatus': MaritalStatus,
            'MonthlyRate': MonthlyRate,
            'NumCompaniesWorked': NumCompaniesWorked,
            'OverTime': OverTime,
            'PerformanceRating': PerformanceRating
            }   
    
    
        #create submit button add to the form
        submit_button = st.form_submit_button(label='Predict')
    

    

        # Convert the input data to a DataFrame 
        input_data = pd.DataFrame(input_data, index=[0])
        # Convert categorical variables to numerical values
        input_data['Department'] = ['Sales', 'Research & Development', 'Human Resources'].index(input_data['Department'][0])
        input_data['EducationField'] = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'].index(input_data['EducationField'][0])
        input_data['Gender'] = ['Male','Female'].index(input_data['Gender'][0])
        input_data['JobRole'] = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Healthcare Representative'].index(input_data['JobRole'][0])
        input_data['MaritalStatus'] = ['Single', 'Married', 'Divorced'].index(input_data['MaritalStatus'][0])
        input_data['OverTime'] = ['Yes', 'No'].index(input_data['OverTime'][0]) 

        # Normalize the input data
        input_data['MonthlyRate'] = (input_data['MonthlyRate']) / (10000 - 5000)
    
     
        # Convert  data to DataFrame
        input_df = pd.DataFrame(input_data)
        input_data.keys()
        # Display the sample data
        st.write(" Data for Prediction:")
        st.write(input_df)
        input_data.keys()

        # Predict the attrition probability
        attrition_probability = model.predict_proba(input_df)[:,1][0]
        # Display the prediction result
        st.write(f"Predicted Attrition Probability: {attrition_probability:.2f}")


        # Display the prediction result
        if attrition_probability > 0.5:
            st.write("The employee is likely to leave the company.")
        else:
            st.write("The employee is likely to stay with the company.")

with tab2:
    # Load the model
    model = joblib.load('employee_performance_model.pkl')

    with st.form(key='input_forms'):
        # Sample data for prediction
        sample_data = {
            'Age': [29],
            'DailyRate': [.9780],
            'DistanceFromHome': [2],
            'JobSatisfaction':[5],
            'MaritalStatus': [2],
            'OverTime': [1], 
            'PercentSalaryHike': [60],
            'StockOptionLevel': [5],
            'YearsAtCompany': [4],
            'YearsInCurrentRole':[7],
            'YearsWithCurrManager':[5]
        }
        # Convert sample data to DataFrame
        sample_df = pd.DataFrame(sample_data)
        sample_data.keys()
        # Display the sample data
        st.write("Sample Data for Prediction:")
        st.write(sample_df)

        # Predict the attrition probability
        performance_probability = model.predict_proba(sample_df)[:, 1][0]
        # Display the prediction result
        st.write(f"Predicted Performance Probability: {performance_probability}")

        # Create new data for prediction
        # Create the input form
        st.title("Employee Performance Prediction")
        # st.write("Please fill in the details below:")
        # Create a form to collect user input
        Age = st.number_input('Age', min_value=18, max_value=100, value=30)
        DailyRate = st.number_input('DailyRate', min_value=0.0, max_value=10000.0, value=5000.0)
        DistanceFromHome = st.number_input('DistanceFromHome', min_value=0, max_value=100, value=5)
        JobSatisfaction = st.number_input('JobSatisfaction', min_value=1, max_value=5, value=3)
        MaritalStatus = st.selectbox('MaritalStatus', options=['Single', 'Married', 'Divorced'])
        OverTime = st.selectbox('OverTime', options=['Yes', 'No'])
        PercentSalaryHike = st.number_input('PercentSalaryHike', min_value=0.0, max_value=100.0, value=10.0)
        StockOptionLevel = st.number_input('StockOptionLevel', min_value=0, max_value=5, value=1)
        YearsAtCompany = st.number_input('YearsAtCompany', min_value=0, max_value=50, value=5)
        YearsInCurrentRole = st.number_input('YearsInCurrentRole', min_value=0, max_value=50, value=5)
        YearsWithCurrManager = st.number_input('YearsWithCurrManager', min_value=0, max_value=50, value=5)
        # Create a form to collect user input
        st.subheader("Input Details")

        input_details = {
            'Age': Age,
            'DailyRate': DailyRate,
            'DistanceFromHome':DistanceFromHome,
            'JobSatisfaction': JobSatisfaction,
            'MaritalStatus': MaritalStatus,
            'OverTime': OverTime,
            'PercentSalaryHike': PercentSalaryHike,
            'StockOptionLevel': StockOptionLevel,
            'YearsAtCompany': YearsAtCompany,
            'YearsInCurrentRole': YearsInCurrentRole,
            'YearsWithCurrManager': YearsWithCurrManager
        }
        #create submit button add to the form
        submit_button = st.form_submit_button(label='Predict')
        
        # # Normalize the input data
        # scaler = MinMaxScaler()
        # Convert the input data to a DataFrame
        input_details = pd.DataFrame(input_details, index=[0])
        # Convert categorical variables to numerical values
        input_details['MaritalStatus'] = ['Single', 'Married', 'Divorced'].index(input_details['MaritalStatus'][0])
        input_details['OverTime'] = ['Yes', 'No'].index(input_details['OverTime'][0])
        ## Normalize the input data
        input_details['DailyRate'] = (input_details['DailyRate']) / (10000 - 0)
        # input_data['PercentSalaryHike'] = (input_data['PercentSalaryHike']) / (100 - 0)
        # input_data['YearsAtCompany'] = (input_data['YearsAtCompany']) / (50 - 0)
        # input_data['YearsInCurrentRole'] = (input_data['YearsInCurrentRole']) / (50 - 0)

    

        # Convert  data to DataFrame
        input_details = pd.DataFrame(input_details)
       
        # Display the sample data
        st.write(" Data for Prediction:")
        st.write(input_details)
       
        
        # Predict the perfromance probability
        pred = model.predict_proba(input_details)[:, 1][0]
        # Display the prediction result
        st.write(f"Predicted Performance Probability: {pred:.2f}")

        # Display the prediction result
        performance_rating = model.predict(input_details)[0]
        st.write(f"Predicted Performance Rating: {performance_rating}")
        # Display the prediction result
        if pred > 0.5:
            st.write("The employee is likely to be a high performer.")
        else:
            st.write("The employee is likely to be a low performer.")



       






