import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

wine = pd.read_csv('csv/winequality.csv', delimiter=';')
print(wine.head())

x = wine.iloc[:, 0:-1].values
y = wine.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
 

z = [1, 2, 3, 4, 5]
r=  [1, 2, 3, 4, 5]
plt.scatter(z, r)
plt.show()
sns.heatmap(wine.corr())


#Using Random Forest Classifier

'''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_jobs=100, random_state=0)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

table = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})

print(table.head())

from sklearn.metrics import accuracy_score
acc_rfc = accuracy_score(y_test, y_predict) * 100
print(
  'The accuracy gotten from applying the Random Forest Classifier Algorithm is',
  round(acc_rfc, 2)
, end='% ')'''

#Using Logistic Regression
'''from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

table = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
print(table.head())
from sklearn.metrics import accuracy_score
acc_lor = accuracy_score(y_test, y_predict) * 100
print(
  'The accuracy gotten from applying the Logistic Regression Algorithm is',
  round(acc_lor, 2)
, end='% ')'''



#Using KMeans Cluster

'''from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.cluster import KMeans

model = KMeans()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

table = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
print(table.head())
from sklearn.metrics import accuracy_score
acc_KM = accuracy_score(y_test, y_predict) * 100
print(
  'The accuracy gotten from applying the KMeans Clustering Algorithm is',
  round(acc_KM, 2)
, end='% ')'''