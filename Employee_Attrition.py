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
            'Age': [37],
            'Department': ['Research & Development'],
            'DistanceFromHome': [2],
            'EducationField': ['Other'],
            'EnvironmentSatisfaction':[4],
            'Gender': ['Male'],
            'JobRole': ['Laboratory Technician'],
            'JobSatisfaction':[3],
            'MaritalStatus': ['Single'],
            'MonthlyRate': [2396],
            'NumCompaniesWorked': [6],
            'OverTime': ['Yes'],
            'PerformanceRating': [3],
            'TotalWorkingYears':[7],
            'PercentSalaryHike':[15]
        }

        
        # Convert the input data to a DataFrame 
        input_data = pd.DataFrame(sample_data, index=[0])
        # Convert categorical variables to numerical values
        input_data['Department'] = ['Sales', 'Research & Development', 'Human Resources'].index(input_data['Department'][0])
        input_data['EducationField'] = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'].index(input_data['EducationField'][0])
        input_data['Gender'] = ['Male','Female'].index(input_data['Gender'][0])
        input_data['JobRole'] = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Healthcare Representative'].index(input_data['JobRole'][0])
        input_data['MaritalStatus'] = ['Single', 'Married', 'Divorced'].index(input_data['MaritalStatus'][0])
        input_data['OverTime'] = ['Yes', 'No'].index(input_data['OverTime'][0]) 
        # Normalize the input data
        scaler = MinMaxScaler()
        # input_data[['Age', 'Department', 'DistanceFromHome', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PerformanceRating', 'TotalWorkingYears',
        #              'PercentSalaryHike']] = scaler.fit_transform(input_data[['Age', 'Department', 'DistanceFromHome', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PerformanceRating', 'TotalWorkingYears', 'PercentSalaryHike']])
        input_data['MonthlyRate'] = (input_data['MonthlyRate']) / (50000 - 0)
        input_data['PercentSalaryHike'] = (input_data['PercentSalaryHike']) / (100 - 0)
        input_data['TotalWorkingYears'] = (input_data['TotalWorkingYears']) / (50 - 0)    
        input_data['DistanceFromHome'] = (input_data['DistanceFromHome']) / (100 - 0)
        input_data['NumCompaniesWorked'] = (input_data['NumCompaniesWorked']) / (10 - 0)
        input_data['PerformanceRating'] = (input_data['PerformanceRating']) / (5 - 1)
        input_data['Age'] = (input_data['Age']) / (100 - 18)
        input_data['JobSatisfaction'] = (input_data['JobSatisfaction']) / (5 - 1)
        input_data['EnvironmentSatisfaction'] = (input_data['EnvironmentSatisfaction']) / (4 - 1)

        # Convert  data to DataFrame
        sample_df = pd.DataFrame(input_data)
        sample_data.keys()
        # Display the sample data
        st.write("Sample Data for Prediction:")
        st.write(sample_df)
        #Predict the attrition probability
        # Predict the attrition probability
        attrition_probability = model.predict_proba(sample_df)[:, 1][0]
        # Display the prediction result
        st.write(f"Predicted Attrition Probability: {attrition_probability:.2f}")
        # Display the prediction result
        if attrition_probability > 0.5:
            st.write("The employee is likely to leave the company.")
        else:
            st.write("The employee is likely to stay with the company.")
        # Display the prediction result
      


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
        EducationField = st.selectbox('EducationField', options=['Life Sciences', 'Medical', 'Marketing', 'Technical Degree','Human Resources', 'Other'])     
        EnvironmentSatisfaction = st.radio('EnvironmentSatisfaction',[1,2,3,4],)
        Gender = st.pills('Gender', options=['Male','Female'],default='Male'),
        JobRole = st.selectbox('JobRole', options=['Sales Executive','Sales Representative', 'Research Scientist', 'Research Director','Laboratory Technician','Manufacturing Director', 'Healthcare Representative','Manager','Human Resources']),
        JobSatisfaction = st.radio('JobSatisfaction',[1,2,3,4]),
        MaritalStatus = st.selectbox('MaritalStatus', options=['Single', 'Married', 'Divorced']),
        MonthlyRate = st.number_input('MonthlyRate', min_value=0.0, max_value=50000.0, value=5000.0),
        NumCompaniesWorked = st.number_input('NumCompaniesWorked', min_value=0, max_value=10, value=2),
        OverTime = st.checkbox('OverTime', value=False)
        if OverTime:
            OverTime = 'Yes'
        else:
            OverTime = 'No'
        PerformanceRating = st.number_input('PerformanceRating', min_value=1, max_value=5, value=3)
        TotalWorkingYears = st.slider('TotalWorkingYears', min_value=0, max_value=50, value=1)
        PercentSalaryHike = st.number_input('PercentSalaryHike', min_value=0.0, max_value=100.0, value=10.0)
        # Create a form to collect user input
        st.subheader("Input Form") 



        input_data = {
            'Age': Age,
            'Department': Department,
            'DistanceFromHome':DistanceFromHome,
            'EducationField': EducationField,
            'EnvironmentSatisfaction': EnvironmentSatisfaction,
            'Gender': Gender,
            'JobRole': JobRole,
            'JobSatisfaction': JobSatisfaction,
            'MaritalStatus': MaritalStatus,
            'MonthlyRate': MonthlyRate,
            'NumCompaniesWorked': NumCompaniesWorked,
            'OverTime': OverTime,
            'PerformanceRating': PerformanceRating,
            'TotalWorkingYears': TotalWorkingYears,
            'PercentSalaryHike': PercentSalaryHike
            }   
    
    
        #create submit button add to the form
        submit_button = st.form_submit_button(label='Predict')  
        # Convert the input data to a DataFrame 
        input_data = pd.DataFrame(input_data, index=[0])
        # Convert categorical variables to numerical values  
        input_data['Department'] = ['Sales', 'Research & Development', 'Human Resources'].index(input_data['Department'][0])  
        input_data['EducationField'] = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree','Human Resources', 'Other'].index(input_data['EducationField'][0])
        input_data['Gender'] = ['Male','Female'].index(input_data['Gender'][0])
        input_data['JobRole'] = ['Sales Executive','Sales Representative', 'Research Scientist', 'Research Director','Laboratory Technician','Manufacturing Director', 'Healthcare Representative','Manager','Human Resources'].index(input_data['JobRole'][0])
        input_data['MaritalStatus'] = ['Single', 'Married', 'Divorced'].index(input_data['MaritalStatus'][0])
        input_data['OverTime'] = ['Yes', 'No'].index(input_data['OverTime'][0]) 
        # Normalize the input data
               
        input_data['MonthlyRate'] = (input_data['MonthlyRate']) / (50000 - 0)
        input_data['PercentSalaryHike'] = (input_data['PercentSalaryHike']) / (100 - 0)
        input_data['TotalWorkingYears'] = (input_data['TotalWorkingYears']) / (50 - 0)
        input_data['DistanceFromHome'] = (input_data['DistanceFromHome']) / (100 - 0)
        input_data['NumCompaniesWorked'] = (input_data['NumCompaniesWorked']) / (10 - 0)
        input_data['PerformanceRating'] = (input_data['PerformanceRating']) / (5 - 1)
        input_data['Age'] = (input_data['Age']) / (100 - 18)
        input_data['JobSatisfaction'] = (input_data['JobSatisfaction']) / (5 - 1)
        input_data['EnvironmentSatisfaction'] = (input_data['EnvironmentSatisfaction']) / (4 - 1)


        # Convert  data to DataFrame
        input_df = pd.DataFrame(input_data)
        input_data.keys()
        # Display the sample data
        st.write(" Data for Prediction:")
        st.write(input_df)
        input_data.keys()
        # Normalize the input data

       
       
        # Predict the attrition probability
        attrition_probability = model.predict_proba(input_df)[:,1][0]
        # Display the prediction result
        st.write(f"Predicted Attrition Probability: {attrition_probability:.2f}")


        # Display the prediction result
        if attrition_probability < 0.5:
            st.write("The employee is likely to leave the company.")
            # Insights and Recommendations
            st.write("Insights and Recommendations:")
            st.write("1. The model indicates that factors such as Age, Job Role, and Job Satisfaction are significant predictors of employee attrition.")
            st.write("2. Employees in certain job roles and with lower job satisfaction levels are more likely to leave the company.")
            st.write("3. The company should focus on improving job satisfaction and addressing concerns related to specific job roles to reduce attrition.")
            st.write("4. Regular employee feedback and engagement surveys can help identify areas for improvement.")
        else:
            st.write("The employee is likely to stay with the company.")
            # Insights and Recommendations for those who stay
            st.write("Insights and Recommendations:")
            st.write("1. The model suggests that employees with higher job satisfaction and performance ratings are more likely to stay.")
            st.write("2. The company should continue to foster a positive work environment and provide opportunities for career growth.")
            st.write("3. Regular performance evaluations and feedback can help maintain employee satisfaction and retention.")
            st.write("4. Consider implementing employee recognition programs to acknowledge and reward high-performing employees.")
        
       






with tab2:
    # Load the model
    model = joblib.load('employee_performance_model.pkl')


    with st.form(key='input_forms'):
        # Sample data for prediction
        sample_data = {
            'Age': [32],
            'Department': ['Research & Development'],
            'Education': [1],
            'JobInvolvement':[1],
            'JobLevel': [1],
            'MonthlyIncome': [3919],
            'PercentSalaryHike':[22],
            'YearsAtCompany':[10],
            'YearsInCurrentRole': [2],
            'YearsWithCurrManager': [7]
        }
        # Convert sample data to DataFrame
        sample_df = pd.DataFrame(sample_data)
        sample_data.keys()
        
        #Convert categorical variables to numerical values
        sample_df['Department'] = ['Sales', 'Research & Development', 'Human Resources'].index(sample_df['Department'][0])
             
      
        # Convert  data to DataFrame
        sample_df = pd.DataFrame(sample_df)
        sample_data.keys()
        # Display the sample data
        st.write("Sample Data for Prediction:")
        st.write(sample_df)


        # Predict the attrition probability
        performance_probability = model.predict(sample_df)[0]   
        # Display the prediction result
        st.write(f"Predicted Performance Probability: {performance_probability}")



        # Create new data for prediction
        # Create the input form
        st.title("Employee Performance Prediction")
        # st.write("Please fill in the details below:")
        # Create a form to collect user input
        Age = st.number_input('Age', min_value=18, max_value=100, value=30)
        #dropdown for department
        # Create a dropdown for department
        Department = st.selectbox('Department', options=['Sales', 'Research & Development', 'Human Resources'])
        # Create a radio button for education
        Education = st.radio('Education',[1,2,3,4,5])
        JobInvolvement = st.radio('JobInvolvement',[1,2,3,4])
        JobLevel = st.number_input('JobLevel', min_value=1, max_value=5, value=2)
        MonthlyIncome = st.number_input('MonthlyIncome', min_value=0.0, max_value=20000.0, value=1000.0)
        PercentSalaryHike = st.number_input('PercentSalaryHike', min_value=0.0, max_value=100.0, value=10.0)
        YearsAtCompany = st.slider('YearsAtCompany', min_value=0, max_value=20, value=1)
        YearsInCurrentRole = st.number_input('YearsInCurrentRole', min_value=0, max_value=40, value=1)  
        YearsWithCurrManager = st.slider('YearsWithCurrManager', min_value=0, max_value=20, value=1)
        

        # Create a form to collect user input
        st.subheader("Input Details")

        input_details = {
            'Age': Age,
            'Department':Department,
            'Education': Education,
            'JobInvolvement': JobInvolvement,
            'JobLevel': JobLevel,
            'MonthlyIncome': MonthlyIncome,
            'PercentSalaryHike': PercentSalaryHike,
            'YearsAtCompany': YearsAtCompany,
            'YearsInCurrentRole': YearsInCurrentRole,
            'YearsWithCurrManager': YearsWithCurrManager
        }
        #create submit button add to the form
        submit_button = st.form_submit_button(label='Predict')
       
        # Convert the input data to a DataFrame
        input_detail = pd.DataFrame(input_details, index=[0])
        #Convert categorical variables to numerical valuess
        #get  the index of the department
        input_detail['Department'] = ['Sales','Research & Development','Human Resources'].index(input_detail['Department'][0])
        # Normalize the input data
        input_detail['MonthlyIncome'] = (input_detail['MonthlyIncome']) / (20000 - 0)
      
        input_detail_df = pd.DataFrame(input_detail) # Convert to DataFrame and keep the first row
        input_detail.keys()
       
        # Display the sample data
        st.write(" Data for Prediction:")
        st.write(input_detail_df)
       
        
       
        # Display the prediction result
        performance_rating = model.predict(input_detail_df)[0]
        st.write(f"Predicted Performance Rating: {performance_rating}")
        # Display the prediction result
        if performance_rating > 3:
            st.write("The employee is likely to be a high performer.")
            # Insights and Recommendations for high performers
            st.write("Insights and Recommendations:")
            st.write("1. The model indicates that employees with higher job involvement and education levels are more likely to be high performers.")
            st.write("2. The company should recognize and reward high-performing employees to maintain their motivation and engagement.")
            st.write("3. Consider providing opportunities for career advancement and skill development to retain high performers.")
            st.write("4. Regular performance evaluations and feedback can help identify areas for improvement and growth.")
        else:
            st.write("The employee is likely to be a low performer.")
            # Insights and Recommendations for low performers
            st.write("Insights and Recommendations:")
            st.write("1. The model suggests that employees with lower job involvement and education levels may need additional support to improve their performance.")
            st.write("2. The company should consider providing training and development programs to help low performers enhance their skills.")
            st.write("3. Regular feedback and performance coaching can help low performers identify areas for improvement and set achievable goals.")
            st.write("4. Consider implementing performance improvement plans to support low performers in their development journey.")
            
       

       

       






