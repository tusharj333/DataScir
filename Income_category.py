# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:05:00 2018

@author: Tushar H. Jagdale
         Data Enthusiast   
"""
import pandas as pd
import numpy as np

train_df = pd.read_csv(r'C:\Users\user\Desktop\Data Science\Projects\Imarticus\python projects\Income Category using Logistic Regression\adult_train.csv',
                        delimiter = ' *, *', engine = 'python')
test_df = pd.read_csv(r'C:\Users\user\Desktop\Data Science\Projects\Imarticus\python projects\Income Category using Logistic Regression\adult_test.csv',
                        delimiter = ' *, *', engine = 'python')
#%%
#Train 
train_df.head(20)         # ? value present in the dataset
print(train_df.shape)     #(32561, 15)
print(train_df.info())
print(train_df.describe(include = "all"))

#Test
test_df.head(20)         # ? value present in the dataset
print(test_df.shape)     #(16281, 15)
print(test_df.info())
print(test_df.describe(include = "all"))



#%% Missing values

print(train_df.isnull().sum())        # No missing value present.
print(test_df.isnull().sum())        # No missing value present.
#%% Counting  '?' values --->>
col = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race',
       'native_country', 'sex', 'income']
print("No. of ? in Train \n")
for value in col:
    print(value, sum(train_df[value] == '?' ))

print("\n No. of ? in Test \n")
for value in col:
    print(value, sum(test_df[value] == '?' ))
# '?' found in  workclass, occupation, native_country of both test and train dataset    
#%% create a copy of Data Frame
train_df_c = pd.DataFrame.copy(train_df)
test_df_c  = pd.DataFrame.copy(test_df)

#%% Replacing ? by mode in categorical columns : workclass, occupation, native_country
col = ['workclass', 'occupation','native_country']

for value in col:
    train_df_c[value].replace(['?'],[train_df_c[value].describe(include = "all")[2]],inplace =True)
    
for value in col:
    test_df_c[value].replace(['?'],[test_df_c[value].describe(include = "all")[2]],inplace = True)
        
   
# Counting '?' values

print("\n No. of ? in Train \n")
for value in col:
    print(value, sum(train_df_c[value] == '?' ))
print("\n No. of ? in Test \n")
for value in col:
    print(value, sum(test_df_c[value] == '?' ))

# Zero '?' value left in both the dataset

#%% #%% COnverting categorical into numerical for both train_df  and test_df
colname =  ['workclass','education', 'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']

# for preprocessing the data

from sklearn import preprocessing

le = {}

for x in colname:
    le[x] = preprocessing.LabelEncoder()  ## assigns numbers to the categories
for x in colname:
    train_df_c[x] = le[x].fit_transform(train_df_c.__getattr__(x))

for x in colname:
    test_df_c[x] = le[x].fit_transform(test_df_c.__getattr__(x))


print(train_df_c.head())     # all values are numerical now
#Check the target variable income 
print(train_df['income'].head())
print(train_df_c['income'].head())    
 
#     <=50K  ---> 0
#     < 50K  ---> 1


print(test_df_c.head())     # all values are numerical now
#Check the target variable income 
print(test_df['income'].head())
print(test_df_c['income'].head())    


        
#%% Split X and Y for train_df

Y = train_df_c.values[:,-1]    # last column ( dependent variable --> Income)
X = train_df_c.values[:,:-1]   # InDependent variable --> all coulumns except Income 

#%% Scaling for train_df
# As the ranges of different independent variable varies a lot, the model may be 
# biased towards some of the higher range independent variables. Distribution is uneven
# to avoid this we need to scale data and distribute evenly -- > mean =0, std = 1 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)                   #Create new values

X = scaler.transform(X)         #fit the new values into original columns

Y = Y.astype(int)

#%% Split into train and test  from train_df only 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 10)
#random_state is set.seed function


#%% Model building --> Logistic Regression
from sklearn.linear_model import LogisticRegression

#create a model
classifier = (LogisticRegression())
#fitting training data to the model
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(list(zip(Y_test, Y_pred)))

#%%  Metric--> confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

confusion_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: \n", confusion_matrix)

print("Classification_report : \n", classification_report(Y_test, Y_pred))
print("Accuracy Score: ",accuracy_score(Y_test, Y_pred))

#%% As type II error (FN- False Negatives) is not acceptable, we will be trying to reduce the value.

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

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cnf = confusion_matrix(Y_test.tolist(), y_pred_class)
print("Confusion Matrix: \n", cnf)

print("Classification_report : \n", classification_report(Y_test.tolist(), y_pred_class))
print("Accuracy Score: ",accuracy_score(Y_test.tolist(), y_pred_class))

#%% Cross Validation

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
""" Here, the model building part is done. Need to predict the income category 
for Test dataset using this model
"""

#%% Testing the model on Test data set 

# Splitting X and Y from test_df_c


X_TEST = test_df_c.values[:,:-1]
Y_TEST = test_df_c.values[:, -1]

#%% Scaling for train_df

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_TEST)                   #Create new values

X_TEST = scaler.transform(X_TEST)         #fit the new values into original columns

Y_TEST = Y_TEST.astype(int)

#%% Prediction with the model builded
#%% Model building --> Logistic Regression
from sklearn.linear_model import LogisticRegression

#create a model
classifier = (LogisticRegression())
#fitting training data to the model
classifier.fit(X_train, Y_train)

Y_PRED_final = classifier.predict(X_TEST)

#%%  Metric--> confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

confusion_matrix = confusion_matrix(Y_TEST, Y_PRED_final)
print("Confusion Matrix: \n", confusion_matrix)

print("Classification_report : \n", classification_report(Y_TEST, Y_PRED_final))
print("Accuracy Score: ",accuracy_score(Y_TEST, Y_PRED_final))

#%% END