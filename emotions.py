import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("ANAD.csv")

print(df.head())

print(df['Emotion '].unique())

plt.figure(figsize = (10, 8))
sns.countplot(df['Emotion '])
plt.show()

print("null values",df.isnull().sum().sum()) # 0


X = df.drop(['name', 'Emotion '], axis = 1) #features
y = df['Emotion '] #labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)

#1 Model - Logistic Regression
m1 = LogisticRegression()
m1.fit(X_train, y_train)
pred1 = m1.predict(X_test)
print(classification_report(y_test, pred1))

# 2 model - Random Forest
grid = {'n_estimators': [10, 50, 100, 300]}
m2 = GridSearchCV(RandomForestClassifier(), grid)
m2.fit(X_train, y_train)
print(m2.best_params_)
pred2 = m2.predict(X_test)
print(classification_report(y_test, pred2))

# 3 Model - Gradient boast
grid = {
    'learning_rate': [0.3, 0.1, 0.5], 
    'n_estimators': [100, 300], 
    'max_depth': [1, 3, 9]
}

m3 = GridSearchCV(GradientBoostingClassifier(), grid, verbose = 2)
m3.fit(X_train, y_train) 
print(m3.best_params_)
pred3 = m3.predict(X_test)
print(classification_report(y_test, pred3))