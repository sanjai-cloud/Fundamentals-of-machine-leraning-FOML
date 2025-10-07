import pandas as pd
import numpy as np
bnotes = pd.read_csv('/content/BankNote_Authentication.csv')
bnotes.head(10)

O/P

x = bnotes.drop('class',axis=1)
y = bnotes['class']
print(x.head(2))
print(y.head(2))

O/P

from sklearn.model_selection import train_test_split
#train_test ratio = 0.2
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.neural_network import MLPClassifier
# activation function : relu
mlp = MLPClassifier(max_iter=500,activation='relu')
mlp.fit(x_train,y_train)
MLPClassifier(max_iter=500)
pred = mlp.predict(x_test)
print(pred)

O/P

from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_test,pred)
print(classification_report(y_test,pred))

O/P

# activation function : logistic
mlp = MLPClassifier(max_iter=500,activation='logistic')
mlp.fit(x_train,y_train)

MLPClassifier(activation='logistic', max_iter=500)
pred = mlp.predict(x_test)
print(pred)

O/P

from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_test,pred)

O/P

print(classification_report(y_test,pred))

O/P

mlp = MLPClassifier(max_iter=500,activation='tanh')
mlp.fit(x_train,y_train)
pred = mlp.predict(x_test)
print(pred)

O/P

from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_test,pred)

O/P

print(classification_report(y_test,pred))

O/P

# activation function : identity
mlp = MLPClassifier(max_iter=500,activation='identity') 
mlp.fit(x_train,y_train)
MLPClassifier(activation='identity', max_iter=500) 
pred = mlp.predict(x_test)
print(pred)

O/P

from sklearn.metrics import classification_report,confusion_matrix 
confusion_matrix(y_test,pred)

O/P

print(classification_report(y_test,pred))

O/P

#train_test ratio = 0.3
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3) 
from sklearn.neural_network import MLPClassifier
# activation function : relu
mlp = MLPClassifier(max_iter=500,activation='relu') 
mlp.fit(x_train,y_train) 
MLPClassifier(max_iter=500)
pred = mlp.predict(x_test) 
print(pred)

O/P

from sklearn.metrics import classification_report,confusion_matrix 
confusion_matrix(y_test,pred)

O/P

print(classification_report(y_test,pred))

O/P

# activation function : logistic
mlp = MLPClassifier(max_iter=500,activation='logistic') 
mlp.fit(x_train,y_train) 
MLPClassifier(max_iter=500,activation='logistic')
pred = mlp.predict(x_test) 
print(pred)
MLPClassifier(max_iter=500,activation='tanh')

O/P

# activation function : tanh
mlp = MLPClassifier(max_iter=500,activation='tanh') 
mlp.fit(x_train,y_train)
pred = mlp.predict(x_test) 
print(pred)

O/P

from sklearn.metrics import classification_report,confusion_matrix 
confusion_matrix(y_test,pred)

O/P

print(classification_report(y_test,pred))

O/P

# activation function : identity
mlp = MLPClassifier(max_iter=500,activation='identity') 
mlp.fit(x_train,y_train) 
MLPClassifier(max_iter=500,activation='identity')
pred = mlp.predict(x_test) 
print(pred)

O/P

from sklearn.metrics import classification_report,confusion_matrix 
confusion_matrix(y_test,pred)

O/P

print(classification_report(y_test,pred))

O/P
