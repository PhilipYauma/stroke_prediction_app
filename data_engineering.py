import pandas as pd,numpy as np
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('new_data_stroke_n.csv')
df.head()
#df.drop(columns = 'ever_married', axis = 1, inplace = True)
df.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)
target = 'stroke'
encode = ['hypertension', 'heart_disease','ever_married','gender','work_type', 'Residence_type', 'smoking_status']

for col in encode:
    dummy = pd.get_dummies(df[col],prefix = col)
    df = pd.concat([df,dummy],axis = 1)
    del df[col]
target_mapper = {'positive_stroke': 1,
                 'negative_stroke': 0}
def target_encode(val):
    return target_mapper[val]

df['stroke'] = df['stroke'].apply(target_encode)

#df.age.astype('float32')
#df.bmi.astype('float32')
#df.avg_glucose_level.astype('float32')
X = df.drop('stroke', axis = 1)
y = df.stroke
clf = RandomForestClassifier()
clf.fit(X,y)
l

import pickle
pickle.dump(clf,open('STK_clf_n.pkl','wb'))
