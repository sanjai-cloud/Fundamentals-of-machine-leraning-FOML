import numpy as np 
import pandas as pd 
from sklearn import svm
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set(font_scale=1.2) 
recipes=pd.read_csv('/content/recipes_muffins_cupcakes.csv') 
recipes.head()
recipes.shape

O/P

sns.lmplot(x='Sugar',y='Flour',data=recipes,hue='Type',palette='Set1',fit_reg=False,scatter_kws={"s":70})

O/P

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
import pandas as pd
import numpy as np
from sklearn import svm

recipes=pd.read_csv('/content/recipes_muffins_cupcakes.csv')

sugar_butter=recipes[['Sugar','Flour']].values
type_label=np.where(recipes['Type']=='Muffin',0,1)
model=svm.SVC(kernel='linear')
model.fit(sugar_butter,type_label)
w=model.coef_[0] #seperating the hyperplane
a = -w[0]/w[1] # calculate a
xx=np.linspace(5,30)
yy=a*xx-(model.intercept_[0]/w[1])
b=model.support_vectors_[0] #plot to seperate hyperplane that pass
yy_down=a*xx+(b[1]-a*b[0])
b=model.support_vectors_[-1]
yy_up=a*xx+(b[1]-a*b[0])
sns.lmplot(x='Sugar',y='Flour',data=recipes,hue='Type',palette='Set1',fit_reg=False,scatter_kws={"s":70})
plt.plot(xx,yy,linewidth=2,color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.show()

O/P

scatter_kws={"s":70}
plt.plot(xx,yy,linewidth=2,color='black')
sns.lmplot(x='Sugar',y='Flour',data=recipes,hue='Type',palette='Set1',fit_reg=False,scatter_kws={"s":70})
plt.plot(xx,yy,linewidth=2,color='black')
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')
plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,-1],s=80,facecolor='none')

O/P

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

x_train,x_test,y_train,y_test = train_test_split(sugar_butter,type_label,test_size=0.2)
model1=svm.SVC(kernel='linear')
model1.fit(x_train,y_train)
pred = model1.predict(x_test)
print(pred)

O/P

print(confusion_matrix(y_test,pred))

O/P

print(classification_report(y_test,pred))

O/P
