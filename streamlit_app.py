import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

# Custom transformer for label encoding categorical columns
class CustomFeaturesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Create 'previous_contact'
        X_new = X.copy()
        X_new['previous_contact'] = (X_new['pdays'] != 999).astype(int)
        X_new.loc[X_new["previous_contact"] == 0, "pdays"] = -1
        
        # Create 'unemployed'
        X_new["unemployed"] = X_new["job"].isin(["student", "retired", "unemployed"]).astype(int)
        
        return X_new

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.encoders = None
    
    def fit(self, X, y=None):
        self.encoders = {
            col: LabelEncoder().fit(X[col].astype(str))
            for col in self.categorical_columns
        }
        return self
    
    def transform(self, X, y=None):
        X_new = X.copy()
        for col, le in self.encoders.items():
            # Handle unknown categories by converting them to string 'unknown'
            X_new[col] = X_new[col].map(lambda s: 'unknown' if s not in le.classes_ else s).astype(str)
            le.classes_ = np.append(le.classes_, 'unknown')
            X_new[col] = le.transform(X_new[col])
        return X_new


# loading the saved model
loaded_model = pickle.load(open('bank_marketing_prediction.sav', 'rb'))

def bank_marketing_prediction(input_data):
    # Correct the feature names to match the training data
    column_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 
                    'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                    'euribor3m', 'nr.employed']  # Use dot notation instead of underscores

    input_df = pd.DataFrame([input_data], columns=column_names)
    prediction = loaded_model.predict(input_df)
    return prediction[0]


def main():
    
    # Giving a title
    st.title('Bank Marketing Prediction Web App')
    
    # Getting input from the user
    age = st.number_input('Age')
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                               'retired', 'self-employed', 'services', 'student', 'technician', 
                               'unemployed', 'unknown'])
    marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox('Education Level', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                                 'illiterate', 'professional.course', 'university.degree', 'unknown'])
    default = st.selectbox('Credit in Default', ['no', 'yes', 'unknown'])
    housing = st.selectbox('Housing Loan', ['no', 'yes', 'unknown'])
    loan = st.selectbox('Personal Loan', ['no', 'yes', 'unknown'])
    contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])
    month = st.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 
                                                'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox('Last Contact Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    campaign = st.number_input('Number of Contacts Performed During this Campaign')
    pdays = st.number_input('Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign')
    previous = st.number_input('Number of Contacts Performed Before this Campaign')
    poutcome = st.selectbox('Outcome of the Previous Marketing Campaign', ['failure', 'nonexistent', 'success'])
    emp_var_rate = st.number_input('Employment Variation Rate')
    cons_price_idx = st.number_input('Consumer Price Index')
    cons_conf_idx = st.number_input('Consumer Confidence Index')
    euribor3m = st.number_input('Euribor 3 Month Rate')
    nr_employed = st.number_input('Number of Employees')

    # Code for prediction
    prediction = ''
    
    # Getting the input data from the user
    if st.button('Bank Marketing Prediction'):
        prediction = bank_marketing_prediction([age, job, marital, education, default, housing, loan,
                                                contact, month, day_of_week, campaign, pdays, previous, 
                                                poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, 
                                                euribor3m, nr_employed])
        
    st.success(prediction)
    
if __name__ == '__main__':
    main()
