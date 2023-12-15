#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:15:50 2023

@author: sneharavi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn import metrics
from collections import defaultdict
from dmba import plotDecisionTree, textDecisionTree
import  pydotplus
from graphviz import Digraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import dmba as dmba
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

os.chdir(r'/Users/sneharavi/Desktop/Quantitaive_Methods/Week_7')
os.getcwd()

df = pd.read_csv('adult.csv')
df.info()
df.isna().sum()
df.duplicated().sum()
sns.countplot(data = df, x='sex', hue ='salary', palette="Set2")

%matplotlib inline
sns.countplot(data = df, x= 'education', hue ='salary', palette="flare")
plt.title('Distribution of Education with Salary')
plt.xticks(rotation = 90)

sns.countplot(data = df, x= 'occupation', hue ='salary', palette="husl")
plt.title('Occupation vs Salary')
plt.xticks(rotation = 90)


%matplotlib inline
sns.countplot(data = df, x= 'salary', palette="Spectral")
plt.title('Distribution of Salary')

sns.histplot(data=df, x='age', hue= 'salary', palette="dark")
plt.title('Distribution of Age with Salary')

adult = df.drop_duplicates()
adult.head()
adult.info()
adult.shape
adult.columns
adult['salary'].unique()
#df.to_excel('adult.xlsx')


#-----------------------------------------------------------------------------

# Convert to ordinal and Nominal Variables

# Education: 0 = Doctorate, 15 = Preschool
adult.education.unique()
ordered_education = [' Doctorate',' Prof-school',' Masters',' Bachelors', ' Assoc-voc' ,' Assoc-acdm',' Some-college',' HS-grad',' 12th' ,' 11th', ' 10th',' 9th',' 7th-8th', ' 5th-6th', ' 1st-4th',' Preschool'] 
adult['edu_order'] = adult.education.astype(CategoricalDtype(categories=ordered_education, ordered=True)).cat.codes
adult.edu_order.value_counts()

# Salary: 1 = >50K, 0 = <=50K
ordered_salary = [' <=50K', ' >50K']
adult['sal_order'] = adult.salary.astype(CategoricalDtype(categories=ordered_salary, ordered=True)).cat.codes
adult.sal_order.value_counts()
adult.salary.value_counts()

# Sex: 1 = Female, 0 = Male
adult.sex.unique()
ordered_sex = [' Female', ' Male']
adult['sex_order'] = adult.sex.astype(CategoricalDtype(categories=ordered_sex, ordered=True)).cat.codes
adult.sex_order.value_counts()

# Race: 0 = Amer-Indian-Eskimo, 1 = Asian-Pac-Islander, 2 = Black, 3 = Other, 4 = White  
race = LabelEncoder()
adult['race_order'] = race.fit_transform(adult['race'])
adult.head()
adult.race_order.value_counts()
adult.race.value_counts()

# 0 = Federal-gov , 1 = Local-gov, 2 = Never-worked, 3 = Private, 4 = Self-emp-inc, 5 = Self-emp-not-inc, 6 = State-gov, 7 = Without-pay, 8 = Unknown
work = LabelEncoder()
adult['work_order'] = work.fit_transform(adult['workclass'])
adult.work_order.value_counts()
adult.workclass.value_counts()

# Occupation
occupation = LabelEncoder()
adult['occu_order'] = occupation.fit_transform(adult['occupation'])
adult.occu_order.value_counts()

#-----------------------------------------------------------------------------

# Set the Predictor and Outcome Variable

predictors = adult[['age','edu_order','race_order','sex_order','work_order','hours_per_week', 'occu_order', 'capital_gain', 'capital_loss']]
outcome = adult['sal_order']
outcome

# Split the Data
x_train, x_test, y_train, y_test = train_test_split(predictors, outcome, test_size=0.20,random_state=42)
x_train.info()
y_train.info()
x_train.shape
x_test.info()

#-----------------------------------------------------------------------------

# Model: RandomForest

# Criterion: Entropy

model_entrop = RandomForestClassifier(n_estimators=35, criterion='entropy', max_depth=10, oob_score=True)
model_entrop.fit(x_train, y_train)
print('Random Forest Entropy Model Training Accuracy Score:' ,round(model_entrop.score(x_train, y_train)*100,3))
print('Random Forest Entropy Model Test Accuracy Score:', round(model_entrop.score(x_test, y_test)*100,3))
# RandomForest: Accuracy
pred = model_entrop.predict(x_test)
pred

cm = confusion_matrix(y_test, pred)
cm
print(cm[0,0])

confusion = dmba.classificationSummary(y_test, pred, class_names=model_entrop.classes_)
labels = ([['True Negative: %s'%cm[0,0], 'False Positive: %s'%cm[0,1]],['False Negative: %s'%cm[1,0], 'True Positive: %s'%cm[1,1]]])
fig, ax = plt.subplots()
sns.color_palette("rocket")
sns.heatmap(cm, annot= labels, fmt = '', cmap="YlGnBu")
plt.title('Confusion Matrix: Random Forest Classifier (Entropy)')

# Roc curve
fpr, tpr, thresholds = roc_curve(y_test, model_entrop.predict_proba(x_test)[:, 0])
roc_entrop = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})

plt.title('Random Forest ROC Curve')
plt.plot(fpr, tpr, color = 'Blue')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Recall')
plt.xlabel('Specificity')
plt.show()

#-----------------------------------------------------------------------------

# Criterion: Gini 

model_gini = RandomForestClassifier(n_estimators=35, criterion='gini', max_depth=10, oob_score=True)
model_gini.fit(x_train, y_train)
model_gini.score(x_train, y_train)
model_gini.score(x_test, y_test)
print('Random Forest Gini Model Training Accuracy Score:' ,round(model_gini.score(x_train, y_train)*100,3))
print('Random Forest Gini Model Test Accuracy Score:', round(model_gini.score(x_test, y_test)*100,3))

# RandomForest: Accuracy
pred_gini = model_gini.predict(x_test)
pred_gini

cm_gini = confusion_matrix(y_test, pred_gini)
cm_gini

dmba.classificationSummary(y_test, pred_gini, class_names=model_gini.classes_)

confusion_gini = dmba.classificationSummary(y_test, pred_gini, class_names=model_gini.classes_)
labels = ([['True Negative: %s'%cm_gini[0,0], 'False Positive: %s'%cm_gini[0,1]],['False Negative: %s'%cm_gini[1,0], 'True Positive: %s'%cm_gini[1,1]]])
fig, ax = plt.subplots()
sns.heatmap(cm_gini, annot= labels, fmt = '', cmap= 'Blues')
plt.title('Confusion Matrix: Random Forest Classifier (Gini)')

#-----------------------------------------------------------------------------

# Feature Importance Entropy

scores = defaultdict(list)
for _ in range(3):
    x_train, x_test, y_train, y_test = train_test_split(predictors, outcome, test_size=0.20,random_state=42)
    model_entrop.fit(x_train, y_train)
    acc = metrics.accuracy_score(y_test, model_entrop.predict(x_test))
    for column in predictors.columns:
        X_t = x_test.copy()
        X_t[column] = np.random.permutation(X_t[column].values)
        shuff_acc = metrics.accuracy_score(y_test, model_entrop.predict(X_t))
        scores[column].append((acc-shuff_acc)/acc)

print('Features sorted by their score:')
print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))

importances = model_entrop.feature_importances_
importances

df1 = pd.DataFrame({'feature': predictors.columns,'Accuracy decrease': [np.mean(scores[column]) for column in predictors.columns],
    'Gini decrease': model_entrop.feature_importances_,
    'Entropy decrease': model_gini.feature_importances_,})

df1 = df1.sort_values('Accuracy decrease')

fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
ax = df1.plot(kind='barh', x='feature', y='Accuracy decrease',legend=False, ax=axes[0], color='pink')
ax.set_ylabel('')
ax = df1.plot(kind='barh', x='feature', y='Gini decrease', legend=False, ax=axes[1], color = 'pink')
ax.set_ylabel('')
ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------

# XGBoost Classifier

X = adult[['age','edu_order','sex_order','work_order','hours_per_week', 'occu_order', 'capital_gain', 'capital_loss']]
Y = adult['sal_order']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
X_train.info()
Y_train.info()
sns.regplot(X_train,y_train)

xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False, subsample = 1, eval_metric = 'error' )
xgb.fit(X_train, Y_train)

print('XG Boost Model Training Accuracy Score:',round((xgb.score(X_train, Y_train))*100,3))
print('XG Boost Model Testing Accuracy Score:',round((xgb.score(X_test, Y_test))*100,3))

xgb_predict = xgb.predict(X_test)
xgb_predict

cm_xgb = confusion_matrix(Y_test, xgb_predict)
cm_xgb

confusion_xg = dmba.classificationSummary(Y_test, xgb_predict, class_names=xgb.classes_)
labels = ([['True Negative: %s'%cm_xgb[0,0], 'False Positive: %s'%cm_xgb[0,1]],['False Negative: %s'%cm_xgb[1,0], 'True Positive: %s'%cm_xgb[1,1]]])
fig, ax = plt.subplots()
sns.heatmap(cm, annot= labels, fmt = '', cmap="BuPu")
plt.title('Confusion Matrix: XGB Classifier')

fpr, tpr, thresholds = roc_curve(Y_test, xgb.predict_proba(X_test)[:, 0])
roc_xgb = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})

plt.title('XGBoost ROC Curve')
plt.plot(fpr, tpr, color = 'green')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Recall')
plt.xlabel('Specificity')
plt.show()

#-----------------------------------------------------------------------------

# Feature Importance XG Boost

scores = defaultdict(list)
for _ in range(3):
    acc = metrics.accuracy_score(Y_test, xgb.predict(X_test))
    for column in X.columns:
        X_t = X_test.copy()
        X_t[column] = np.random.permutation(X_t[column].values)
        shuff_acc = metrics.accuracy_score(Y_test, xgb.predict(X_t))
        scores[column].append((acc - shuff_acc) / acc)

# Sort features based on importance scores
feature_importance = sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)

# Print feature importance
print('Features sorted by their importance score:')
print(feature_importance)

importances = xgb.feature_importances_

df1 = pd.DataFrame({'feature': X.columns, 'Importance': importances})

df1 = df1.sort_values('Importance')

plt.figure(figsize=(8, 5))
plt.barh(df1['feature'], df1['Importance'], color='lightblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('XGBoost Model Feature Importance')
plt.show()

#-----------------------------------------------------------------------------

#Model Comparison

models_df = pd.DataFrame()
models_df['Observations_entrop'] = ['Above 50K' if x == 1 else 'Below 50K' for x in y_test]
models_df['entropy_prediction'] = ['Above 50K' if sal_order == 1 else 'Below 50K' for sal_order in model_entrop.predict(x_test)]
models_df['Observations_XGB'] = ['Above 50K' if x == 1 else 'Below 50K' for x in Y_test]
models_df['xgb_prediction'] = ['Above 50K' if sal_order == 1 else 'Below 50K' for sal_order in xgb_predict]
models_df['prob_entropy'] = model_entrop.predict_proba(x_test)[:, 0]
models_df['prob_xgb'] = xgb.predict_proba(X_test)[:, 0]
print(models_df.head(10))
#models_df.to_excel('prediction.xlsx')


