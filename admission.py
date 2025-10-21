import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

st.title('Graduate Admission Predictor')
st.write("This app uses multiple inputs to predict the probability of admission to grad school")

st.image('admission.jpg', width = 400)

rf_pickle = open('reg_admission.pickle', 'rb')
reg = pickle.load(rf_pickle)
rf_pickle.close()

df = pd.read_csv('Admission_Predict.csv')
st.sidebar.header('Enter your profile details')
gre = st.sidebar.number_input("GRE Score", min_value= df['GRE Score'].min(), max_value= df['GRE Score'].max(),step=1)
toefl = st.sidebar.number_input("TOEFL Score", min_value= df['TOEFL Score'].min(), max_value= df['TOEFL Score'].max(),step=1)
cgpa = st.sidebar.number_input("CGPA", min_value= df['CGPA'].min(), max_value= df['CGPA'].max(),step=0.01)
exp = st.sidebar.selectbox("Research Experieince", ["Yes", "No"])
ur = st.sidebar.slider('University Rating', min_value= df['University Rating'].min(), max_value= df['University Rating'].max(),step=1)
sop = st.sidebar.slider('Statement of Purpose (SOP)', min_value= df['SOP'].min(), max_value= df['SOP'].max(),step=0.1)
lor = st.sidebar.slider('Letter of Recruitment', min_value= df['LOR'].min(), max_value= df['LOR'].max(),step=0.1)

exp_yes, exp_no = 0, 0
if exp == "Yes":
    exp_yes = 1
elif exp == "No":
    exp_no = 1

if st.sidebar.button("Predict"):
    new_prediction = reg.predict([[gre, toefl, cgpa, exp_yes, exp_no, ur, sop, lor]])
    prediction_chance = new_prediction[0]
    st.subheader("Prediction Admission Chance...")
    st.write(f"{round(prediction_chance*100, 2)}%")


## tabs

st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])


with tab1:
     st.write("### Feature Importance")
    # st.image('feature_imp.svg')
     st.caption("Features used in this prediction are ranked by relative importance.")

 # Tab 3: Confusion Matrix
with tab2:
     st.write("### Confusion Matrix")
   #  st.image('confusion_mat.svg')
     st.caption("Confusion Matrix of model predictions.")

 # Tab 4: Classification Report
with tab3:
     st.write("### Classification Report")
     report_df = pd.read_csv('Admission_Predict.csv', index_col = 0).transpose()
     st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
     st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

