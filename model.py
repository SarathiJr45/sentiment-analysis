import pandas as pd
import sklearn
import numpy as np
import pickle

df=pd.read_csv('BankNote_Authentication.csv')

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
print(score)

import pickle
pickle.dump(model,open('model.pkl','wb'))