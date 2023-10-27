# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:10:33 2023

@author: rania

Key:
    survival - survival of the disaster
    pclass - ticket class (1st, 2nd, 3rd)
    sex - sex of the person
    age - age in years
    sibsp - siblings/spouses
    parch - parents/children
    ticket - ticket number
    fare - passenger fare
    cabin - cabin number
    embarked - point of Embarkation (Southampton, Cherbourg, Queenstown)
    
"""

######################
# Data Loading

from pandas import read_csv
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_set = read_csv("train.csv")
test_set = read_csv("test.csv")

# Getting an Idea

train_set.head(10)

"""
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S
5            6         0       3  ...   8.4583   NaN         Q
6            7         0       1  ...  51.8625   E46         S
7            8         0       3  ...  21.0750   NaN         S
8            9         1       3  ...  11.1333   NaN         S
9           10         1       2  ...  30.0708   NaN         C
"""

train_set.describe()

"""
       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200
"""

description_train = train_set.describe()
description_test = test_set.describe()

######################
# Preprocessing


# Nulls

train_set.isnull().sum()
test_set.isnull().sum()

# Heatmap analysis of nulls

sns.heatmap(train_set.isnull(), cbar = False) # cbar not necessary here since this is a binary isnull or is not null
plt.title('Missing Data Matrix')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()

# Fare

plt.hist(train_set['Fare'], alpha = 0.7, log = True)
plt.grid(True)
plt.title('Log Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()


# Age

plt.hist(train_set['Age'], alpha = 0.7)
plt.grid(True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# Let's visualize the Age nulls

train_set['AgeNulls'] = train_set['Age'].isnull()

sns.countplot(data = train_set, x = 'AgeNulls')
plt.title('Train Set - Age Column Nulls')
plt.xlabel('Null Values')
plt.ylabel('Null Count')
plt.xticks([0,1], ['Non-null', 'Null'])
plt.show()

train_set.drop('AgeNulls', axis = 1, inplace = True)

# Let's deal with the Name column. Let's use titles to get the mean of ages to impute NaNs in Age col

train_set['Title'] = train_set['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_counts = train_set['Title'].value_counts()

"""
Mr          517
Miss        182
Mrs         125
Master       40
Dr            7
Rev           6
Mlle          2
Major         2
Col           2
Countess      1
Capt          1
Ms            1
Sir           1
Lady          1
Mme           1
Don           1
Jonkheer      1
Name: Title, dtype: int64
"""

title_counts.plot(kind='bar')
plt.title('Value Counts of Title')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Should I use the mean or median age to decide what to impute?

plt.figure()
sns.boxplot(x = train_set['Age']) #using SNS here so I don't have to handle NaNs
plt.xlabel('Age')
plt.title('Boxplot of Age')
plt.show()

plt.figure()
sns.histplot(data = train_set, x = 'Age', bins = 20)
sns.rugplot(data = train_set, x = 'Age', height = 0.02, color = 'r', alpha = 0.5)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()

# This shows that we have a few outliers. Because of this, median might be better to impute
# rather than mean (which is sensitive to outliers).

# Z-scores to verify outliers

age_data = train_set['Age']

z_scores = (age_data - np.mean(age_data)) / np.std(age_data)

thresh = 2

outliers = np.abs(z_scores) > thresh

print("Percentage of Outliers in Age column: ", (np.sum(outliers) / len(outliers)) * 100)

# Percentage of Outliers in Age column:  3.254769921436588

# Let's look at the median ages for each title

train_set.groupby(['Title']).median()['Age']

"""
Title
Capt        70.0
Col         58.0
Countess    33.0
Don         40.0
Dr          46.5
Jonkheer    38.0
Lady        48.0
Major       48.5
Master       3.5
Miss        21.0
Mlle        24.0
Mme         24.0
Mr          30.0
Mrs         35.0
Ms          28.0
Rev         46.5
Sir         49.0
"""

# Let's also look at the mean ages for each title:
    
train_set.groupby(['Title']).mean()['Age']

"""
Title
Capt        70.000000
Col         58.000000
Countess    33.000000
Don         40.000000
Dr          42.000000
Jonkheer    38.000000
Lady        48.000000
Major       48.500000
Master       4.574167
Miss        21.773973
Mlle        24.000000
Mme         24.000000
Mr          32.368090
Mrs         35.898148
Ms          28.000000
Rev         43.166667
Sir         49.000000
Name: Age, dtype: float64
"""

age_median = train_set.groupby(['Title']).median()['Age']
age_mean = train_set.groupby(['Title']).mean()['Age']

plt.figure(figsize = (15, 10))
sns.barplot(x = age_median.index, y = age_median.values, yerr = np.abs(age_mean - age_median), capsize = 4)
plt.title('Comparison of Median and Mean Ages by Title')
plt.xlabel('Title')
plt.ylabel('Age')
plt.show()


# Let's plot the distributions of age across the titles and see if we have outliers:
    
sns.boxplot(x = 'Title', y = 'Age', data = train_set)
plt.xticks(rotation = 45)
plt.show()

sns.violinplot(x = 'Title', y = 'Age', data = train_set)
plt.xticks(rotation = 45)
plt.show()

# Ultimately, there is such a slight difference between these two, that I will choose to stick with imputing
# the median for missing age values. The median is still more robust against any present outliers, and while those
# outliers don't seem to be so significant as to affect the dataset (this is a pretty small dataset overall), I still
# think choosing the median here would work well, especially since the mean isn't so far off.

#Okay, now I need to use the training data to impute these Ages into any of those NaNs for test and train sets

median_ages = train_set.groupby(['Title']).median()['Age']

train_set['Age'] = train_set.apply(lambda i: median_ages[i['Title']] if pd.isna(i['Age'])
                                   else i['Age'], axis = 1)

# I need to do the same thing to the test set, BUT only using the train_set to avoid data leakage

test_set['Title'] = test_set['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


test_set['Age'] = test_set.apply(lambda i: median_ages[i['Title']] if pd.isna(i['Age'])
                                   and i['Title'] in median_ages.index else i['Age'], axis = 1)

# Cabin has way too many nulls, so I am going to drop the column. I will also drop 
# Ticket, Title, and Name now that I am done with working on that imputation.

train_set.drop(columns = ['PassengerId', 'Cabin', 'Ticket', 'Name', 'Title'], inplace = True)
test_set.drop(columns = ['PassengerId', 'Cabin', 'Ticket', 'Name', 'Title'], inplace = True)

# Now I can deal with dropping any NaNs from both test & train sets:
    
train_set.isnull().sum()
test_set.isnull().sum()    
    
train_set.dropna(inplace=True)
test_set.dropna(inplace=True)

# Categorical Data

# Both Embarked and Sex contain categories that are not ordinally related, and so onehot would be best here

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse = False)
cols_encode = ['Sex', 'Embarked']

train_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)

# Encode the Sex and Embarked Cols in Train Set

train_encoded = pd.DataFrame(encoder.fit_transform(train_set[cols_encode]))
train_encoded.columns = encoder.get_feature_names_out(cols_encode)

print("Row count", train_encoded.shape[0]) #row count 889, that is correct

train_set = pd.concat([train_set, train_encoded], axis = 1)
train_set.drop(['Sex', 'Embarked'], axis = 1, inplace = True)

train_set.isnull().sum() #0 nulls

# Encode the Sex and Embarked Cols in the Test Set

test_encoded = pd.DataFrame(encoder.fit_transform(test_set[cols_encode]))
test_encoded.columns = encoder.get_feature_names_out(cols_encode)

print("Row count", test_encoded.shape[0]) #row count 417, that is correct

test_set = pd.concat([test_set, test_encoded], axis = 1)
test_set.drop(['Sex', 'Embarked'], axis = 1, inplace = True)

test_set.isnull().sum() #0 nulls

'''
Added this in Summer '23 for data vis class, so I can use tableau
for the cleaned training dataset

train_set.to_csv('cleaned_train.csv', index = False)
'''

######################
# Separate Features, Response for Feature Selection & Prediction

X = train_set.drop(['Survived'], axis = 1)
y = train_set['Survived']
Xnames = X.columns

models = []


######################
# EDA

plt.figure()
train_set.hist(figsize = (15, 11))
plt.show()

description_train = train_set.describe()

"""
Quick interperet: 
    some right skewed cols: fare, parch, sibsp
    age is almost gaussian
    passengerid is chaos, drop
    mostly, people did not survive
    mostly, people were male
    most of the people embarked on southampton, then cherbourg, then queensland
"""


######################
# Transformation

# I need to drop the OneHotEncoded features as transforming them is unnecessary.
X1 = X.drop(['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q','Embarked_S'], axis = 1)
X1names = X1.columns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

n_scaler = Normalizer().fit(X1)
normalized_X = n_scaler.transform(X1)

norm_df = pd.DataFrame(normalized_X, columns = X1names)
norm_df['Survived'] = y
norm_describe = norm_df.describe()
norm_df.hist(figsize=(15,11))
plt.show

# That skews everything, or maintains the skew, so probably not a good choice.

s_scaler = StandardScaler().fit(X1)
standard_X = s_scaler.transform(X1)

standard_df = pd.DataFrame(standard_X, columns = X1names)
standard_df['Survived'] = y
standard_describe = standard_df.describe()
standard_df.hist(figsize=(15,11))
plt.show()

# That has a slight improvement in some columns, and helps with Pclass and Age. Will choose to standardize.

# Let's recombine features

transformed_df = pd.concat([standard_df, X[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q','Embarked_S']]], axis = 1)

# And split the set again using that transformed X

X = transformed_df.drop(['Survived'], axis = 1)
y = transformed_df['Survived']
Xnames = transformed_df.columns


# I need to transform the test set to ensure consistency between train & test, and make sure my model will perform appropriately on unseen data

X_test = test_set.drop(['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q','Embarked_S'], axis = 1)

scaled = StandardScaler().fit(X_test)
standard = scaled.transform(X_test)

standard_test = pd.DataFrame(standard, columns = X1names)
standard_test = pd.concat([standard_test, test_set[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q','Embarked_S']]], axis = 1)

# And update test_set with that; yes, I can do that in one step, but prefer to separate just in case:
#test_set = standard_test


######################
# Correlation Analysis

from pandas.plotting import scatter_matrix

# Correlation Analysis of pre-transformed data

plt.figure()
corMat = train_set.corr(method='pearson')

sns.heatmap(corMat, square = True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("Correlation Matrix using Heat Map")
plt.show()

plt.figure()
scatter_matrix(train_set, figsize=(15,11))
plt.show()

# Corr. analysis of post-transformed data

plt.figure()
corMat = transformed_df.corr(method='pearson')

sns.heatmap(corMat, square = True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("Correlation Matrix Of Transformed Data using Heat Map")
plt.show()

plt.figure()
scatter_matrix(transformed_df, figsize=(15,11))
plt.show()

"""
Sex_Female and Survived positively correlated
Sex_Male and Survived negatively correlated
"""
sns.barplot(x='Sex_female', y='Survived', data = transformed_df)

sns.barplot(x='Pclass', y='Survived', data = train_set)

# Note that Pclass might not need to have been transformed.


######################
# Feature Selection

"""
To reduce the scope of the project, I've decided not to complete feature selection. I'll leave
this section as present so I can comment on those reasons later. There are also only so many
features after what we dropped, so let's go ahead and see what the results are.

"""

######################
# Classification

from sklearn.metrics import accuracy_score

# Leaving the following out; if I want the real y from the test set, I would need to submit to the competition
# so I will use the training set to create a split.
#X_train = X
#y_train = y
#X_test = standard_test
#X_train.shape, y_train.shape, X_test.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=42)

# Out[327]: ((889, 10), (889,), (417, 10))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)

lr_prob = lr.predict_proba(X_test)

lr.score(X_test, y_test)

# 0.8129251700680272

models.append(("Logistic Regression", lr))

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth = 2, random_state = 1).fit(X_train, y_train)

tree.score(X_test, y_test)

# 0.7959183673469388

models.append(("Decision Tree Classifier", tree))

# Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 500, random_state = 1).fit(X_train, y_train)

rfc.score(X_test, y_test)

# 0.782312925170068

models.append(("Random Forest Classifier", rfc))

# Linear SVC

from sklearn.svm import LinearSVC

lin_SVC = LinearSVC(C=10.0, loss = 'hinge', random_state = 1, max_iter = 10000).fit(X_train, y_train)

lin_SVC_pred = lin_SVC.predict(X_test)

lin_SVC.score(X_test, y_test)

# 0.7959183673469388

models.append(("Linear SVC", lin_SVC))

# SVC

from sklearn.svm import SVC

SVC = SVC(kernel = 'rbf', C = 10.0, random_state = 1, max_iter = 10000).fit(X_train, y_train)

SVC_pred = SVC.predict(X_test)

SVC.score(X_test, y_test)

#Out[348]: 0.8095238095238095

models.append(("SVC", SVC))

# MLP Classifier

from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier(random_state = 1, max_iter = 3000).fit(X_train, y_train)

MLP_prob = MLP.predict_proba(X_test)

MLP_pred = MLP.predict(X_test)

MLP.score(X_test, y_test)

# 0.8027210884353742

models.append(("MLP Classifier", MLP))

######################
# K-Fold Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle = True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})"
    print(msg)

"""
Logistic Regression: 0.799 (0.035)
Decision Tree Classifier: 0.769 (0.035)
Random Forest Classifier: 0.804 (0.021)
Linear SVC: 0.786 (0.038)
SVC: 0.813 (0.032)
MLP Classifier: 0.816 (0.036)
"""

# Let's try to visualize the results

for i in range(len(names)):
    plt.plot(range(1,11), results[i], label = names[i])
plt.title('Model Performance Comparisons')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

######################
# Classification Reports

from sklearn.metrics import classification_report

models_again = [lr, tree, rfc, lin_SVC, SVC, MLP]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'LinearSVC', 'SVC', 'MLP']

for model, name in zip(models_again, model_names):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    print(name)
    print(report)

"""
Logistic Regression
              precision    recall  f1-score   support

           0       0.86      0.84      0.85       184
           1       0.74      0.77      0.76       110

    accuracy                           0.81       294
   macro avg       0.80      0.80      0.80       294
weighted avg       0.81      0.81      0.81       294

Decision Tree
              precision    recall  f1-score   support

           0       0.77      0.97      0.86       184
           1       0.90      0.51      0.65       110

    accuracy                           0.80       294
   macro avg       0.84      0.74      0.75       294
weighted avg       0.82      0.80      0.78       294

Random Forest
              precision    recall  f1-score   support

           0       0.83      0.83      0.83       184
           1       0.71      0.71      0.71       110

    accuracy                           0.78       294
   macro avg       0.77      0.77      0.77       294
weighted avg       0.78      0.78      0.78       294

LinearSVC
              precision    recall  f1-score   support

           0       0.84      0.84      0.84       184
           1       0.73      0.73      0.73       110

    accuracy                           0.80       294
   macro avg       0.78      0.78      0.78       294
weighted avg       0.80      0.80      0.80       294

SVC
              precision    recall  f1-score   support

           0       0.82      0.90      0.85       184
           1       0.79      0.66      0.72       110

    accuracy                           0.81       294
   macro avg       0.81      0.78      0.79       294
weighted avg       0.81      0.81      0.81       294

MLP
              precision    recall  f1-score   support

           0       0.82      0.88      0.85       184
           1       0.77      0.68      0.72       110

    accuracy                           0.80       294
   macro avg       0.79      0.78      0.78       294
weighted avg       0.80      0.80      0.80       294

Going to focus on Logistic Regression, Random Forest, and SVC for hyperparams optimizing. However, look at MLP!
It's right alongside these algorithms. Because I don't have enough experience to test out this model,
I will focus on the above three, but it is worthy to note that MLP could achieve a higher accuracy.
"""

######################
# Algorithm Comparison

fig = plt.figure(figsize=(15,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


######################
# Hyperparameter Optimization

hyp_models = []

#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Logistic Regression

lr_param_grid = {'C': [0.0001,0.001,0.1,1, 10, 100],
                 'penalty': ['l1', 'l2'],
                 'random_state': [42]}

lr = LogisticRegression(solver='liblinear')

rs_lr = GridSearchCV(lr, lr_param_grid, cv = 5).fit(X_train, y_train)

rs_lr.best_params_

# {'C': 0.1, 'penalty': 'l2'}

best_lr = rs_lr.best_estimator_

lr_acc = best_lr.score(X_test, y_test)

print(lr_acc)

# 0.8163265306122449

hyp_models.append(('Logistic Regression RSCV', rs_lr)) 


# Random Forest

rf_param_grid = {'n_estimators': [5, 10, 15, 20, 25, 30, 35],
              'max_depth': [3, 5, 7, 9, 11, 13],
              'random_state': [42],
              'class_weight': ['balanced']}

rf = RandomForestClassifier()

rf_rs = GridSearchCV(rf, rf_param_grid).fit(X_train, y_train)

rf_rs.best_params_

# {'class_weight': 'balanced','max_depth': 11,'n_estimators': 20,'random_state': 42}

best_rf = rf_rs.best_estimator_

rf_acc = best_rf.score(X_test, y_test)

print(rf_acc)

# 0.8197278911564626

hyp_models.append(('Random Forest RSCV', rf_rs))


# SVC

svc_param_grid = {'C': [0.0001,0.001,0.1,1, 10, 100],
                  'kernel': ['linear', 'rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto'],
                  'random_state': [42]}

svc = SVC()

svc_rs = GridSearchCV(svc, svc_param_grid).fit(X_train, y_train)

best_svc = svc_rs.best_estimator_

svc_rs.best_params_

# {'C': 10, 'gamma': 'scale', 'kernel': 'rbf', 'random_state': 42}

svc_acc = best_svc.score(X_test, y_test)

print(svc_acc)

# 0.8095238095238095

hyp_models.append(('SVC RSCV', svc_rs))



######################
# Confusion Matrices

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

rs_lr_pred = rs_lr.predict(X_test)
rf_rs_pred = rf_rs.predict(X_test)
svc_rs_pred = svc_rs.predict(X_test)

cm_lr = confusion_matrix(y_test, rs_lr_pred)
ConfusionMatrixDisplay(confusion_matrix = cm_lr).plot()
print("Confusion Matrix for Log Regression", cm_lr)

cm_rf = confusion_matrix(y_test, rf_rs_pred)
ConfusionMatrixDisplay(confusion_matrix = cm_rf).plot()
print("Confusion Matrix for Random Forest", cm_rf)

cm_svc = confusion_matrix(y_test, svc_rs_pred)
ConfusionMatrixDisplay(confusion_matrix = cm_svc).plot()
print("Confusion Matrix for Linear SVC", cm_svc)

# Random Forest looks to be the best here, but let's do another precision/recall

hyp_names = ['Logistic Regression', 'Random Forest', 'SVC']
hyp_mod = [rs_lr, rf_rs, svc_rs]

for model, name in zip(hyp_mod, hyp_names):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    print(name)
    print(report)


"""
Logistic Regression
              precision    recall  f1-score   support

           0       0.84      0.87      0.86       184
           1       0.77      0.73      0.75       110

    accuracy                           0.82       294
   macro avg       0.81      0.80      0.80       294
weighted avg       0.81      0.82      0.82       294

Random Forest
              precision    recall  f1-score   support

           0       0.83      0.88      0.85       184
           1       0.77      0.71      0.74       110

    accuracy                           0.81       294
   macro avg       0.80      0.79      0.80       294
weighted avg       0.81      0.81      0.81       294

SVC
              precision    recall  f1-score   support

           0       0.82      0.90      0.85       184
           1       0.79      0.66      0.72       110

    accuracy                           0.81       294
   macro avg       0.81      0.78      0.79       294
weighted avg       0.81      0.81      0.81       294

"""

# According to this, either Random Forest or Logistic Regression would be ideal predictive models
# although these are all quite low in terms of accuracy for a decent ML model.

######################
# K-Fold Cross Validation After Hyperparameter Optimization

hyp_results = []
k_names = []
scoring = 'accuracy'

for name, model in hyp_models:
    kfold = KFold(n_splits=10, random_state=7, shuffle = True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    hyp_results.append(cv_results)
    k_names.append(name)
    msg = f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})"
    print(msg)

"""
Logistic Regression RSCV: 0.803 (0.038)
Random Forest RSCV: 0.828 (0.025)
SVC RSCV: 0.825 (0.039)
"""


for i in range(len(k_names)):
    plt.plot(range(1,11), hyp_results[i], label = k_names[i])
plt.title('Model Performance Comparisons')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


######################
# Algorithm Comparison

# After Hyperparameter Search
fig = plt.figure(figsize=(10,5))
fig.suptitle('Algorithm Comparison After Hyp')
ax = fig.add_subplot(111)
plt.boxplot(hyp_results)
ax.set_xticklabels(k_names)
plt.show()

"""
Shows that RF is best here, since LR has that outlier issue and the RF algo
is better at classifying correctly
"""