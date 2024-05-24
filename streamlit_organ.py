import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the datasets with error handling
try:
    donors_df = pd.read_csv('donors.csv')
    recipients_df = pd.read_csv('recipients.csv')
    matches_df = pd.read_csv('matches.csv')
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Please ensure the file is in the correct directory.")
    st.stop()

# Merge the donors and recipients data with the matches data
matches_df = matches_df.merge(donors_df, on='donor_id', suffixes=('_donor', '_recipient'))
matches_df = matches_df.merge(recipients_df, on='recipient_id', suffixes=('_donor', '_recipient'))

# Define the features and target variable
X = matches_df[['age_donor', 'gender_donor', 'blood_type_donor', 'organ_type', 'height_donor', 'weight_donor',
                'age_recipient', 'gender_recipient', 'blood_type_recipient', 'organ_needed', 'height_recipient',
                'weight_recipient', 'waiting_time_days']]

y = matches_df['match_success']

# Convert categorical variables to numeric
labelencoder = LabelEncoder()
X['gender_donor'] = labelencoder.fit_transform(X['gender_donor'])
X['blood_type_donor'] = labelencoder.fit_transform(X['blood_type_donor'])
X['gender_recipient'] = labelencoder.fit_transform(X['gender_recipient'])
X['blood_type_recipient'] = labelencoder.fit_transform(X['blood_type_recipient'])
X['organ_type'] = labelencoder.fit_transform(X['organ_type'])
X['organ_needed'] = labelencoder.fit_transform(X['organ_needed'])

# Fill any missing values with the mean
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit app
st.title('Organ Donor-Recipient Matching System')

st.header('Donor Information')
donor_age = st.number_input('Age', min_value=0, max_value=100, value=30)
donor_gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
donor_blood_type = st.selectbox('Blood Type', ('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'))
donor_organ_type = st.selectbox('Organ Type', ('Kidney', 'Liver', 'Heart', 'Lung', 'Pancreas'))
donor_height = st.number_input('Height (cm)', min_value=100, max_value=250, value=175)
donor_weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)

st.header('Recipient Information')
recipient_age = st.number_input('Age', min_value=0, max_value=100, value=40)
recipient_gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
recipient_blood_type = st.selectbox('Blood Type', ('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'))
recipient_organ_needed = st.selectbox('Organ Needed', ('Kidney', 'Liver', 'Heart', 'Lung', 'Pancreas'))
recipient_height = st.number_input('Height (cm)', min_value=100, max_value=250, value=170)
recipient_weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=65)
waiting_time_days = st.number_input('Waiting Time (days)', min_value=0, max_value=3650, value=100)

if st.button('Predict Match Success'):
    # Prepare input data
    input_data = pd.DataFrame({
        'age_donor': [donor_age],
        'gender_donor': [labelencoder.fit_transform([donor_gender])[0]],
        'blood_type_donor': [labelencoder.fit_transform([donor_blood_type])[0]],
        'organ_type': [labelencoder.fit_transform([donor_organ_type])[0]],
        'height_donor': [donor_height],
        'weight_donor': [donor_weight],
        'age_recipient': [recipient_age],
        'gender_recipient': [labelencoder.fit_transform([recipient_gender])[0]],
        'blood_type_recipient': [labelencoder.fit_transform([recipient_blood_type])[0]],
        'organ_needed': [labelencoder.fit_transform([recipient_organ_needed])[0]],
        'height_recipient': [recipient_height],
        'weight_recipient': [recipient_weight],
        'waiting_time_days': [waiting_time_days]
    })

    # Predict the match success
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success('Match Successful!')
    else:
        st.error('Match Unsuccessful!')
