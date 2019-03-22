import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
%matplotlib inline

###### loading files ######
train = pd.read_csv(r'C:\Users\sharon.pan\Desktop\python\Project -\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\sharon.pan\Desktop\python\Project -\Titanic\test.csv')

train.isnull().sum()

# preprocessing lable
le = preprocessing.LabelEncoder()

###### Training Set ######
# handling Sex
train['Sex'] = le.fit_transform(train['Sex'])

# handling Embarked
train['Embarked'].fillna('S',inplace=True)
train['Embarked'] = le.fit_transform(train['Embarked'])

# handling Name with Title
for i in range(train.shape[0]):
    train.loc[i,'Title'] = re.search(r"(?<=,).*?(?=\.)",train.loc[i,'Name'])[0].strip()
train['Title_1'] = le.fit_transform(train['Title'])
# np.unique(np.array(Title))
# Col = 陸軍上校
# Don = Sir(Spanish)
# Jonkheer = 鄉紳(紳士階級中較低的,男性尊稱)
# Major = 少校
# Master = 紳士
# Mlle = 小姐(未婚)
# Mme = Madame 太太 = Mrs
# Rev = 牧師
# the Countess = 伯爵夫人

# family size
train['Fam_size'] = train['SibSp'] + train['Parch']

# Age Group
train['Age_Group'] = np.floor_divide(train['Age'],10)

####### predicting those missing values in Age #######
train3 = train.copy()
train3withAge = train3[train3['Age_Group'].isnull() == False]
train3NOAge = train3[train3['Age_Group'].isnull() == True]

train3withAgetrain = train3withAge.iloc[:,[4,5,7]]
train3withAgelabel = train3withAge.iloc[:,8]
train3AgeSet, test3AgeSet, train3AgeLabel, test3AgeLabel = train_test_split(train3withAgetrain,train3withAgelabel,random_state = 10, test_size = 0.25)

#Decision Tree
#Finding best min_samples_split
GiniAcc3 = []
for i in range(2,30):
    GiniModel3 = DecisionTreeClassifier(min_samples_split =i)
    GiniModel3.fit(train3AgeSet, train3AgeLabel)
    GiniPred3 = GiniModel3.predict(test3AgeSet)
    GiniAcc3.append([i,accuracy_score(y_true = test3AgeLabel, y_pred = GiniPred3)])

GiniAcc3 = pd.DataFrame(GiniAcc3,columns=['num','acc'])
GiniAcc3[GiniAcc3['acc'] == GiniAcc3['acc'].max()]

# fill in the missing Age
# features: SibSp, Parch, Title_1
Age_model = DecisionTreeClassifier(min_samples_split = 2)
Age_model.fit(train3AgeSet, train3AgeLabel)
Age_predict = Age_model.predict(train3NOAge.iloc[:,[4,5,7]])
for i in range(train3.shape[0]):
    if np.isnan(train3.loc[i,'Age_Group']) == True:
        train3.loc[i,'Age_Group'] = Age_model.predict(train3.iloc[i,[4,5,7]].values.reshape(-1,3))
        
###### Test set ######
# handling Sex
test['Sex'] = le.fit_transform(test['Sex'])

# handling Embarked
test['Embarked'] = le.fit_transform(test['Embarked'])

# handling Name with Title
for i in range(test.shape[0]):
    test.loc[i,'Title'] = re.search(r"(?<=,).*?(?=\.)",test.loc[i,'Name'])[0].strip()
test['Title_1'] = le.fit_transform(test['Title'])
print(test['Title'].unique())

# np.unique(np.array(Title))

# family size
test['Fam_size'] = test['SibSp'] + test['Parch']

# Missing Age & convert to Age Group
test['Age_Group'] = np.floor_divide(test['Age'],10)
for i in range(test.shape[0]):
    if np.isnan(test.loc[i,'Age_Group']) == True:
        test.loc[i,'Age_Group'] = Age_model.predict(test.iloc[i,[5,6,12]].values.reshape(-1,3))           
