import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pickle as pkl


st.write("""#Stroke Prediction Application""")
st.sidebar.header("User input Features")
st.sidebar.markdown("""[Example CSV file](https://www.kaggle.com/search?q=stroke)""")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type = ["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('gender',('Male','Female'))
        age = st.sidebar.slider('age',0, 82	, 43)
        hypertension = 	st.sidebar.selectbox('hypertension',('positive','negative'))
        heart_disease = st.sidebar.selectbox('heart_disease',('positive','negative'))
        work_type = st.sidebar.selectbox('work_type',('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
        ever_married = st.sidebar.selectbox('ever_married',('Yes','No'))
        Residence_type = st.sidebar.selectbox('Residence_type',('Urban', 'Rural'))
        avg_glucose_level = st.sidebar.slider('avg_glucose_level',55,271,106)
        bmi = st.sidebar.slider('bmi',10,97,28)
        smoking_status = st.sidebar.selectbox('smoking_status',('formerly smoked', 'never smoked', 'smokes', 'Unknown'))

        data = {'gender' : gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'work_type': work_type,
                'ever_married': ever_married,
                'Residence_type': Residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status}
        features = pd.DataFrame(data, index = [0])
        return features;
    input_df = user_input_features()

dfw = pd.read_csv('new_data_stroke_n.csv')
dfw.drop('Unnamed: 0', axis = 1, inplace = True)
#dfw.drop(columns = 'ever_married', axis = 1, inplace = True)
#df1 = dfw.astype({'age':'int8',
                 # 'avg_glucose_level':'float16',
                  #'bmi_new':'float16'})
dfw[['avg_glucose_level', 'bmi']] = dfw[['avg_glucose_level', 'bmi']].astype('float16')
dfw['age'] = dfw['age'].astype('int16')
stroke_row = dfw
stroke = stroke_row.drop(columns = ['stroke'])
df = pd.concat([input_df,stroke], axis = 0)

encode = ['hypertension','heart_disease','gender', 'work_type', 'ever_married', 'Residence_type', 'smoking_status']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

st.subheader('User input feature')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Waiting CSV file to be uploaded.Currently using example input parameters')
    st.write(df)

load_clf = pkl.load(open('STK_clf_n.pkl','rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
stroke_pre = np.array(['likely positive','likely negative'])
st.write(stroke_pre[prediction])
st.subheader('Prediction probability')
st.write(prediction_proba)
