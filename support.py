import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset = pd.read_csv(r"D:\projects\onlinefraud.csv" , nrows=100000)
dataset = dataset.drop(["step" , "nameOrig" , "nameDest" ,"isFlaggedFraud"] , axis=1)
X = dataset.iloc[: , :-1]
y = dataset.iloc[: , -1]

encode = LabelEncoder()
X["type"] = encode.fit_transform(X["type"])

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0)

classifier = SVC(kernel='rbf' , random_state=0)
classifier.fit(X_train , y_train)

y_pred = classifier.predict(X_test)
cf = confusion_matrix(y_test , y_pred)
print(cf)
acc = accuracy_score(y_test , y_pred)
print(acc)