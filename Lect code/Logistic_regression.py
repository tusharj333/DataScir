# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 08:32:44 2018

@author: Tushar DSP_6
Logistic Regression 
"""

import pandas as pd
adult_df = pd.read_csv(r'C:\Users\admin\Desktop\Tushar\Python\Logistic regression\adult_data.csv'
                 , header = None, delimiter = ' *, *', engine = 'python')
#%%
print(adult_df.head())
adult_df.shape
adult_df.info()
adult_df.describe(include = "all")
#%% Column names
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

print(adult_df.head())

#%%
adult_df.isnull().sum()

for value in ['workclass','education', 'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income'] : 
    print (value, ":", sum(adult_df[value] == '?'))

#%% create a copy of Data Frame
adult_df_rev = pd.DataFrame.copy(adult_df)   
print(adult_df_rev.describe(include = 'all'))    

#%%
## Replacing ? by mode in categorical columns : workclass, occupation, native_country
for value in ['workclass', 'occupation','native_country']:
    adult_df_rev[value].replace(['?'], [adult_df_rev.describe(include = "all")[value][2]], inplace = True) 

#%%
#print(adult_df_rev.describe(include = "all")) 
print(adult_df.head(20))
print(adult_df_rev.head(20))

#%% COnverting categorical into numerical
colname =  ['workclass','education', 'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']

# for preprocessing the data

from sklearn import preprocessing

le = {}

for x in colname:
    le[x] = preprocessing.LabelEncoder()  ## assigns numbers to the categories
for x in colname:
    adult_df_rev[x] = le[x].fit_transform(adult_df_rev.__getattr__(x))
    
#%%
adult_df_rev.head()

# 0 ----> <=50K
# 1 ----> > 50K

#%% 
Y = adult_df_rev.values[:,-1]    # last column ( dependent variable --> Income)
X = adult_df_rev.values[:,:-1]   # InDependent variable --> all coulumns except Income 

#%% As the ranges of different independent variable varies a lot, the model may be 
#biased towards some of the higher range independent variables. Distribution is uneven
# to avoid this we need to scale data and distribute evenly -- > mean =0, std = 1 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)                   #Create new values

X = scaler.transform(X)         #fit the new values into original columns

Y = Y.astype(int)
#%% Split into train and test 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 10)
#random_state is set.seed function

#%%
from sklearn.linear_model import LogisticRegression

#create a model
classifier = (LogisticRegression())
#fitting training data to the model
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(list(zip(Y_test, Y_pred)))

#%%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

confusion_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: \n", confusion_matrix)

print("Classification_report : \n", classification_report(Y_test, Y_pred))
print("Accuracy Score: ",accuracy_score(Y_test, Y_pred))

#%% As type II error is not acceptable, we will be trying to reduce the value.

# Store the predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)


y_pred_class = []
for value in y_pred_prob[:,0]:
    if value < 0.7:
        y_pred_class.append(1)
    else :
        y_pred_class.append(0)
        
#print(y_pred_class)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cnf = confusion_matrix(Y_test.tolist(), y_pred_class)
print("Confusion Matrix: \n", cnf)

print("Classification_report : \n", classification_report(Y_test.tolist(), y_pred_class))
print("Accuracy Score: ",accuracy_score(Y_test.tolist(), y_pred_class))

#%%
classifier = (LogisticRegression())

#performimg k fold cross validation 
from sklearn import cross_validation
kfold_cv = cross_validation.KFold(n=len(X_train), n_folds = 10)
print(kfold_cv)

# running the model using scoring metric as accuracy 
kfold_cv_result = cross_validation.cross_val_score(estimator = classifier, X = X_train, y = Y_train,
                                            scoring = "accuracy", cv = kfold_cv)
print(kfold_cv_result)

#finding the mean 
print(kfold_cv_result.mean())

""" 
# Here in this case our models accuracy is 0.8227.. and by kfold technique ..
# the accuracy is 0.824324 ... there is not much difference , our model was good.

if there was a drastic difference in the model's accuracy then we need
 to select the model by Kfold technique

"""
#%%
# Kfold best model selection 

for train_value, test_value in kfold_cv:
    classifier.fit(X_train[train_value], Y_train[train_value]).predict_proba(X_train[test_value])
    
Y_pred_kf = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cnf_kf = confusion_matrix(Y_test, Y_pred_kf)
print("Confusion Matrix: \n", cnf_kf)

print("Classification_report : \n", classification_report(Y_test.tolist(), Y_pred_kf))
print("Accuracy Score: ",accuracy_score(Y_test.tolist(), Y_pred_kf))

#%%
 



 
 

