import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("digital_eye.csv")
d1=df.drop("Schimers1Lefteye",axis=1)
d1=d1.drop("Schimers2Lefteye",axis=1)
d1=d1.drop("Schimers1righteye",axis=1)
d1=d1.drop("Schimers2righteye",axis=1)
columns_list = d1.columns.tolist()
d2=df.drop(columns=columns_list)
x=d1
y=d2
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=42)
x_train=d1
y_train=d2
scaler= StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test= scaler.fit_transform(x_test)
model=RandomForestClassifier(n_estimators=20)
model.fit(x_train, y_train)
filename = 'digital_eye.pkl'
pickle.dump(model, open(filename, 'wb'))