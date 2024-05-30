import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('bank-additional.csv', delimiter=';')
    return data

data = load_data()

# Preprocess the data
def preprocess_data(data):
    target_column = 'y'
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                           'day_of_week', 'poutcome']
    numerical_columns = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                         'euribor3m', 'nr.employed']

    # Encode the target variable
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Create a column transformer to apply transformations to specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', LabelEncoder(), categorical_columns),  # Label encode categorical columns
            ('num', StandardScaler(), numerical_columns)  # Standardize numerical columns
        ]
    )

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)

# Train the model
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_classifier)
])
pipeline_rf.fit(X_train, y_train)

# Display the web app
st.title('Bank Marketing Prediction App')

# Sidebar for user inputs
st.sidebar.header('Enter Customer Information')

# Collect user inputs
age = st.sidebar.number_input('Age', min_value=18, max_value=100, step=1)
job = st.sidebar.selectbox('Job', data['job'].unique())
marital = st.sidebar.selectbox('Marital', data['marital'].unique())
education = st.sidebar.selectbox('Education', data['education'].unique())
default = st.sidebar.selectbox('Default', data['default'].unique())
housing = st.sidebar.selectbox('Housing', data['housing'].unique())
loan = st.sidebar.selectbox('Loan', data['loan'].unique())
contact = st.sidebar.selectbox('Contact', data['contact'].unique())
month = st.sidebar.selectbox('Month', data['month'].unique())
day_of_week = st.sidebar.selectbox('Day of Week', data['day_of_week'].unique())
duration = st.sidebar.number_input('Last Contact Duration', min_value=0, step=1)
campaign = st.sidebar.number_input('Number of Contacts Performed', min_value=1, step=1)
previous = st.sidebar.number_input('Number of Contacts Before Campaign', min_value=0, step=1)
pdays = st.sidebar.number_input('Days Since Last Contact', min_value=0, step=1)
previous_outcome = st.sidebar.selectbox('Previous Campaign Outcome', data['poutcome'].unique())
emp_var_rate = st.sidebar.number_input('Employment Variation Rate', min_value=-3.0, max_value=3.0, step=0.1)
cons_price_idx = st.sidebar.number_input('Consumer Price Index', min_value=92.0, max_value=95.0, step=0.1)
cons_conf_idx = st.sidebar.number_input('Consumer Confidence Index', min_value=-50.0, max_value=-25.0, step=0.1)
euribor3m = st.sidebar.number_input('Euribor 3-Month Rate', min_value=0.0, max_value=5.0, step=0.1)
nr_employed = st.sidebar.number_input('Number of Employees', min_value=4900, max_value=5300, step=10)

# Prepare user input for prediction
user_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'previous': [previous],
    'pdays': [pdays],
    'poutcome': [previous_outcome],
    'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx],
    'cons.conf.idx': [cons_conf_idx],
    'euribor3m': [euribor3m],
    'nr.employed': [nr_employed]
})

# Make prediction
prediction = pipeline_rf.predict(user_data)

# Display prediction result
st.subheader('Prediction')
if prediction[0] == 1:
    st.write('The customer is likely to subscribe to the term deposit.')
else:
    st.write('The customer is unlikely to subscribe to the term deposit.')

