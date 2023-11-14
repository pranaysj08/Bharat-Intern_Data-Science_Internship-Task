#!/usr/bin/env python
# coding: utf-8

# DATA SCIENCE INTERN @ BHARAT INTERN
# 
# AUTHOR - PRANAY S JADHAV
# 
# TASK NO. - 01 TITANIC DATASET CLASSIFICATION 
# 
# AIM : TO BUILD A CLASSIFICATION MODEL USING ML ALGORITHMS THAT PREDICTS THE SURVIVAL RATE OF PASSANGERS.
# 
# DATASET INFO : 
#     
# One of history's most notorious shipwrecks is the Titanic. During her inaugural journey, on April 15, 1912, 
# the Titanic capsized due to an iceberg collision, resulting in the deaths of 1502 passengers and crew members. 
# Improved ship safety standards resulted from this extraordinary disaster that startled the world.
# 
# It appears that certain groups of people had a higher chance of living than others, 
# even though survival did involve some degree of luck.
# 
# In order to complete this challenge, you must develop a prediction model that responds to the query, 
# "What kinds of people were more likely to survive?" making use of passenger data, such as name, age, gender, etc.
# 
# THE DATASET IS AVAILABLE ON KAGGLE : https://www.kaggle.com/competitions/titanic/data

# In[73]:


# IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[4]:


# LOADING THE DATASET

titanic = pd.read_csv("C:\\Users\\PRANAY\\Downloads\\Titanic\\titanic_dataset.csv")
titanic


# In[5]:


# FIRST 5 ROWS

titanic.head()


# In[6]:


# LAST 5 ROWS

titanic.tail()


# In[7]:


# NO. OF COLUMNS & ROWS

titanic.shape


# In[8]:


# Columns

titanic.columns


# # Data Processing & Data Cleaning

# In[9]:


#CHECKING DATA TYPES

titanic.dtypes


# In[10]:


# CHECKING DUPLICATE

titanic.duplicated().sum()


# In[11]:


# CHECKING NULL VALUES

nv = titanic.isna().sum().sort_values(ascending=False)
nv = nv[nv>0]
nv


# In[12]:


# CHECKING % OF MISSING VALUES IN RESPECTIVE COLUMNS

titanic.isnull().sum().sort_values(ascending=False)*100/len(titanic)
     


# In[13]:


# DROPPING COLUMN = CABIN AS IT HAS HIGHER NO. OF NULL VALUES

titanic.drop(columns = 'Cabin', axis = 1, inplace = True)
titanic.columns


# In[14]:


# FILLING NULL VALUES

# FILLING NULL VAUES IN AGE WITH MEAN OF AGE COLUMN

titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)


# In[15]:



titanic['Age']


# In[16]:


titanic['Age'].isnull().sum()


# In[17]:


# FILLING NULL VAUES OF EMBARKED COLUMN WITH MODE VALUES OF EMBARKED COLUMN

titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)


# In[18]:


titanic['Embarked']


# In[19]:


titanic['Embarked'].isnull().sum()


# In[20]:


# CONFIRMING NULL VALUES AGAIN

titanic.isna().sum()


# In[21]:


# FINDING UNIQUE VALUES IN COLUMNS

titanic[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']].nunique().sort_values()
     


# In[22]:


titanic['Survived'].unique()


# In[23]:


titanic['Sex'].unique()


# In[24]:


titanic['Pclass'].unique()


# In[25]:


titanic['SibSp'].unique()


# In[26]:


titanic['Parch'].unique()


# In[27]:


titanic['Embarked'].unique()


# # Dropping unnecessary columns

# In[28]:


titanic.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)
titanic.columns


# In[29]:


# DATASET INFO

titanic.info()


# In[30]:


# NUMERICAL INFO OF COLUMNS

titanic.describe()


# In[31]:


# CATEGORICAL INFO OF COLUMNS

titanic.describe(include='O')


# # Visualizing the Dataset

# In[32]:


d1 = titanic['Sex'].value_counts()
d1


# In[33]:


sns.countplot(x=titanic['Sex'])
plt.show()


# In[34]:


# % DISTRIBUTION OF SEX COLUMN

plt.figure(figsize=(5,5))
plt.pie(d1.values,labels=d1.index,autopct='%.2f%%')
plt.legend()
plt.show()


# In[35]:


# DISTRIBUTION OF SURVIVED BASED ON SEX

# In Sex (0 represents female and 1 represents male)

sns.countplot(x=titanic['Sex'],hue=titanic['Survived'])


# In[36]:


# DISTRIBUTION OF EMBARKED BASED ON SEX

sns.countplot(x=titanic['Embarked'],hue=titanic['Sex'])
plt.show()


# In[37]:


# PLOTTING COUNT PLOT FOR COLUMN PCLASS

sns.countplot(x=titanic['Pclass'])
plt.show()


# In[38]:


# DISTRIBUTION OF PCLASS BASED ON SEX

sns.countplot(x=titanic['Pclass'],hue=titanic['Sex'])
plt.show()


# In[39]:


# AGE DISTRIBUTION GRAPH

sns.kdeplot(x=titanic['Age'])
plt.show()


# In[40]:


# PLOTTING COUNTPLOT FOR SURVIVED PASSENGERS

print(titanic['Survived'].value_counts())
sns.countplot(x=titanic['Survived'])
plt.show()


# In[41]:


# DISTRIBUTION OF PARCH FOR SURVIVED 

sns.countplot(x=titanic['Parch'],hue=titanic['Survived'])
plt.show()


# In[42]:


# DISTRIBUTION OF SIBSP FOR SURVIVED

sns.countplot(x=titanic['SibSp'],hue=titanic['Survived'])
plt.show()


# In[43]:


# DISTRIBUTION OF EMBARKED BASED ON SURVIVED PASSENGERS

sns.countplot(x=titanic['Embarked'],hue=titanic['Survived'])
plt.show()


# In[44]:


# VISUALIZATION BASED ON AGE FOR SURVIVED PASSENGERS

sns.kdeplot(x=titanic['Age'],hue=titanic['Survived'])
plt.show()


# In[45]:


# PLOTTING HISTOGRAM FOR DATASET

titanic.hist(figsize=(10,10))
plt.show()


# In[77]:


# CHECKING FOR ANY OUTLIERS USING BOXPLOT

sns.boxplot(data=titanic)
plt.show()


# In[ ]:


# SHOWING CORRELATION

titanic.corr()


# In[47]:


# CORRELATION HEATMAP

sns.heatmap(titanic.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[48]:


# PLOTTING PAIRPLOT OF THE DATASET

sns.pairplot(titanic)
plt.show()


# In[49]:


# SURVIVED PASSENGERS 

titanic['Survived'].value_counts()


# In[50]:


sns.countplot(x=titanic['Survived'])
plt.show()


# # Label Encoding

# In[51]:


from sklearn.preprocessing import LabelEncoder

# CREATING AN INSTANCE OF LABEL ENCODER

le = LabelEncoder()

# APPLYING LABEL ENCODING TO EACH CATEGORICAL COLUMN

for column in ['Sex','Embarked']:
    titanic[column] = le.fit_transform(titanic[column])

titanic.head()

# SEX COLUMN

# 0 represents Female
# 1 represents Male

# EMBARKED COLUMN

# 0 represents C
# 1 represents Q
# 2 represents S


# In[52]:


# IMPORTING ML LIBRARIES

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[53]:


# SELECTING DEPENDENT AND INDEPENDENT FEATURES


# In[54]:


cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = titanic[cols]
y = titanic['Survived']
print(x.shape)
print(y.shape)
print(type(x))  # DataFrame
print(type(y))  # Series


# In[55]:


x.head()


# In[56]:


y.head()


# # Train_Test_Split

# In[57]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # CREATING FUNCTION TO COMPUTE
# 
# # 1. CONFUSION MATRIX
# 
# # 2. CLASSIFICATION REPORT
# 
# # 3. TO GENERATE TRAINING & TESTING SCORE i.e. ACCURACY

# In[58]:


def cls_eval(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(ytest,ypred))

def mscore(model):
    print('Training Score',model.score(x_train,y_train))  # Training Accuracy
    print('Testing Score',model.score(x_test,y_test))     # Testing Accuracy


# In[59]:


# BUILDING THE LOGISTIC REGRESSION MODEL

lr = LogisticRegression(max_iter=1000,solver='liblinear')
lr.fit(x_train,y_train)


# In[60]:


# COMPUTING TRAING AND TESTING SCORE

mscore(lr)


# In[61]:


# GENERATING PREDICTION

ypred_lr = lr.predict(x_test)
print(ypred_lr)


# In[62]:


# EVALUATING THE MODEL

cls_eval(y_test,ypred_lr)
acc_lr = accuracy_score(y_test,ypred_lr)
print('Accuracy Score',acc_lr)


# In[63]:


# BUILDING KNN CLASSIFIER MODEL

knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)


# In[64]:


# COMPUTING TRAINING & TESTING SCORE

mscore(knn)


# In[65]:


# GENERATING PREDICTION

ypred_knn = knn.predict(x_test)
print(ypred_knn)


# In[66]:


# EVALUATING THE MODEL

cls_eval(y_test,ypred_knn)
acc_knn = accuracy_score(y_test,ypred_knn)
print('Accuracy Score',acc_knn)


# In[78]:


# BUILDING THE SUPPORT VECTOR CLASSIFIER MODEL

svc = SVC(C=1.0)
svc.fit(x_train, y_train)


# In[79]:


# COMPUTING TRAINING AND TESTING SCORE

mscore(svc)


# In[80]:


# GENERATING PREDICTIONS

ypred_svc = svc.predict(x_test)
print(ypred_svc)


# In[81]:


# EVALUATING THE MODEL

cls_eval(y_test,ypred_svc)
acc_svc = accuracy_score(y_test,ypred_svc)
print('Accuracy Score',acc_svc)


# In[82]:


# BUILDING THE RANDOM FOREST CLASSIFIER MODEL

rfc=RandomForestClassifier(n_estimators=80,criterion='entropy',min_samples_split=5,max_depth=10)
rfc.fit(x_train,y_train)


# In[83]:


# COMPUTING THE TRAINING AND TESTING SCORE

mscore(rfc)


# In[84]:


# GENERATING PREDICTIONS

ypred_rfc = rfc.predict(x_test)
print(ypred_rfc)


# In[85]:


# EVALUATING THE MODEL

cls_eval(y_test,ypred_rfc)
acc_rfc = accuracy_score(y_test,ypred_rfc)
print('Accuracy Score',acc_rfc)


# In[86]:


# BUILDING THE DECISION TREE MODEL

dt = DecisionTreeClassifier(max_depth=5,criterion='entropy',min_samples_split=10)
dt.fit(x_train, y_train)


# In[87]:


# COMPUTING TRAINING AND TESTING SCORES

mscore(dt)


# In[88]:


# GENERATING PREDICTIONS

ypred_dt = dt.predict(x_test)
print(ypred_dt)


# In[89]:


# EVALUATING THE MODEL

cls_eval(y_test,ypred_dt)
acc_dt = accuracy_score(y_test,ypred_dt)
print('Accuracy Score',acc_dt)


# In[90]:


# BUILDING THE ADABOOST MODEL

ada_boost  = AdaBoostClassifier(n_estimators=80)
ada_boost.fit(x_train,y_train)


# In[91]:


# COMPUTING THE TRAINING AND TESTING SCORE

mscore(ada_boost)


# In[92]:


# GENERATING PREDICTIONS

ypred_ada_boost = ada_boost.predict(x_test)


# In[93]:


# EVALUATING THE MODEL

cls_eval(y_test,ypred_ada_boost)
acc_adab = accuracy_score(y_test,ypred_ada_boost)
print('Accuracy Score',acc_adab)


# In[94]:


models = pd.DataFrame({
    'Model': ['Logistic Regression','knn','SVC','Random Forest Classifier','Decision Tree Classifier','Ada Boost Classifier'],
    'Score': [acc_lr,acc_knn,acc_svc,acc_rfc,acc_dt,acc_adab]})

models.sort_values(by = 'Score', ascending = False)


# In[95]:


colors = ["blue", "green", "red", "yellow","orange","purple"]

sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=models['Model'],y=models['Score'], palette=colors )
plt.show()


# Decision Tree Classifier Model shows the Highest Accuracy.

# In[ ]:




