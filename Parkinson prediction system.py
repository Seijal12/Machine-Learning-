#!/usr/bin/env python
# coding: utf-8

# # Parkinson prediction system
# 

# In[ ]:





# # Importing the Dependencies 

# In[39]:


import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# # Data collection & Analysis

# In[3]:


df= pd.read_csv("Parkinsson disease.csv")



# In[4]:


df


# In[40]:


#Printing first 5 rows of the dataframe
df.head()


# In[42]:


#number of rows and columns in the dataframe
df.shape


# In[5]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

df


# In[6]:


#getting more information of the dataset
df.info()


# In[7]:


#statiscal measures
df.describe()


# In[8]:


#checking for missing values in each column
df.isnull().sum()


# In[9]:


#distribution of target variable
df['status'].value_counts()


# 1---------> Parkinson's Positive
# 0---------> Healthy

# In[10]:


#grouping the data based on the target variable
df.groupby('status').mean()


# # Data Pre-Processing

# Separating the features and Target

# In[43]:


X = df.drop(columns=['name','status'],axis = 1)
Y =df['status']


# In[14]:


print(X)


# In[15]:


print(Y)


# Spilitting the data to training data and test data
# 

# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2,random_state=2)


# In[17]:


print (X.shape,X_train.shape,X_test.shape)


# Data Standardization
# 

# In[18]:


ss = StandardScaler()


# In[19]:


ss.fit(X_train)


# In[20]:


X_train =ss.transform(X_train)
X_test =ss.transform(X_test)


# In[21]:


print(X_train)


# In[22]:


print(X_test)


# # Model Training

# Support Vector Machine Model
# 

# In[23]:


model = svm.SVC(kernel = 'linear')


# In[25]:


model.fit(X_train , Y_train)


# # Model Evaluation

# Accuracy score

# In[27]:


X_train_pred = model.predict(X_train)
train_data_acc = accuracy_score(Y_train , X_train_pred)


# In[28]:


print("accuracy of training data:", train_data_acc)


# In[29]:


X_test_pred = model.predict(X_test)
test_data_acc = accuracy_score(Y_test , X_test_pred)


# In[30]:


print("accuracy of testing data:", test_data_acc)


# # Building a Predictive System

# In[45]:


input_data =(116.848,217.552,99.503,0.00531,0.00005,0.0026,0.00346,0.0078,0.01795,0.163,0.0081,0.01144,0.01756,0.02429,0.01179,22.085,0.663842,0.656516,-5.198864,0.206768,2.120412,0.252404)

#changing input data into numpy array

input_data_np = np.asarray(input_data)

#reshape the numpy array

input_data_re = input_data_np.reshape(1,-1)

#standardize the data

s_data = ss.transform(input_data_re)

prediction = model.predict(s_data)

print(prediction)

if(prediction[0] ==0):
    print("Negative , No Parkinson's Found")
else:
    print("Positive, Parkinson found")


# In[ ]:




