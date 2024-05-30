import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
@st.cache_data
def load_data():
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    target_var = bank_marketing.data.targets
    # UCIMLRepo says to remove this, since the outcome of this data is not known before the call, so its useless for prediction
    features = X.drop('duration', axis=1)
    features=features.drop('balance',axis=1)
    #Feature Engineering to remove the 999 values in pdays, and create a new binary class instead( "contacted")
    features['contacted'] = features['pdays'].apply(lambda x: 0 if x == 999 else 1)
    features=features.drop('pdays',axis=1)

    return target_var, features


@st.cache_data
def initialize_and_train_model(features,target_var):
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = features.select_dtypes(include=['object']).columns
    numeric_prep = Pipeline([
    ('median_imputer', SimpleImputer(strategy='median')),
    ('standard_scaler', StandardScaler())])
    categorical_prep = Pipeline([
    ('fill_missing', SimpleImputer(strategy='constant', fill_value='missing')),
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))])
    data_preprocessor = ColumnTransformer([
    ('numeric', numeric_prep, numeric_cols),
    ('categorical', categorical_prep, categorical_cols)])
    model_pipeline = Pipeline([
    ('data_preprocessing', data_preprocessor),
    ('ada_classifier', AdaBoostClassifier(n_estimators=50,learning_rate=0.1,random_state=42))])
    model_pipeline.fit(features,target_var)

    return model_pipeline


def main():
    st.title("Bank Marketing Prediction App")

    # Load data and model
    target_var, features = load_data()
    model = initialize_and_train_model(features, target_var)

    # Show a snippet of the dataset
    if st.checkbox('Show data sample'):
        st.write(features.head())

    # Predictions
    st.subheader('Make a Prediction')

    # Numeric Inputs
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    campaign = st.number_input('Campaign', min_value=0, value=1)
    previous = st.number_input('Previous', min_value=0, value=0)
    emp_var_rate = st.number_input('Emp Var Rate', value=0.0)
    cons_price_idx = st.number_input('Cons Price Idx', value=0.0)
    cons_conf_idx = st.number_input('Cons Conf Idx', value=0.0)
    euribor3m = st.number_input('Euribor 3m', value=0.0)
    nr_employed = st.number_input('Nr Employed', value=0.0)
    contacted = st.number_input('Contacted', min_value=0, max_value=1, value=0)

    # Categorical Inputs
    job = st.selectbox('Job', options=features['job'].unique())
    marital = st.selectbox('Marital', options=features['marital'].unique())
    education = st.selectbox('Education', options=features['education'].unique())
    default = st.selectbox('Default', options=features['default'].unique())
    housing = st.selectbox('Housing', options=features['housing'].unique())
    loan = st.selectbox('Loan', options=features['loan'].unique())
    contact = st.selectbox('Contact', options=features['contact'].unique())
    month = st.selectbox('Month', options=features['month'].unique())
    day_of_week = st.selectbox('Day of Week', options=features['day_of_week'].unique())
    poutcome = st.selectbox('Poutcome', options=features['poutcome'].unique())

    # Predict button
    if st.button('Predict'):
        # Create a DataFrame from the input features
        input_data = pd.DataFrame([[age, job, marital, education, default, housing, loan, contact,
                                    month, day_of_week, campaign, previous, poutcome, emp_var_rate,
                                    cons_price_idx, cons_conf_idx, euribor3m, nr_employed, contacted]],
                                  columns=['age', 'job', 'marital', 'education', 'default', 'housing', 
                                           'loan', 'contact', 'month', 'day_of_week', 'campaign', 
                                           'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 
                                           'cons.conf.idx', 'euribor3m', 'nr.employed', 'contacted'])

        # Make a prediction
        prediction = model.predict(input_data)
        st.write(f"The prediction is: {prediction[0]}")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()

