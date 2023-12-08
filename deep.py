import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

dataset = pd.read_csv(r"D:\projects\onlinefraud.csv")
dataset = dataset.drop(["step" , "nameOrig" , "nameDest" ,"isFlaggedFraud"] , axis=1)
X = dataset.iloc[: , :-1]
y = dataset.iloc[: , -1]

encode = LabelEncoder()
X["type"] = encode.fit_transform(X["type"])

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 10 , activation='relu'))
ann.add(tf.keras.layers.Dense(units = 5 , activation='relu'))
ann.add(tf.keras.layers.Dense(units = 10 , activation='relu'))

ann.add(tf.keras.layers.Dense(units = 1 , activation='sigmoid'))

ann.compile(optimizer='adam' , loss = 'binary_crossentropy' , metrics=['accuracy'])

ann.fit(X_train , y_train , batch_size = 75000 , epochs = 20)

y_pred = ann.predict(X_test)

cf = mean_absolute_error(y_pred , y_test)
print(cf)