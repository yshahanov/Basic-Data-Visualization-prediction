#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv("C:/Users/Van Shahanov/Desktop/Programming/Python/CSV DOCS/500_Person_Gender_Height_Weight_Index.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[6]:


from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# In[11]:


gender = LabelEncoder()
data['Gender'] = gender.fit_transform(data['Gender'])


# In[12]:


data


# In[15]:


bins = (-1,0,1,2,3,4,5)
health = ['malnourished','underweight', 'fit', 'slightly overweight', 'overweight', 'extremely overweight']
data['Index'] = pd.cut(data['Index'], bins = bins, labels = health)


# In[16]:


data['Index']


# In[21]:


data['Gender'].value_counts()


# In[23]:


sns.countplot(data['Index'], height = 5, aspect = 3)


# In[26]:


sns.relplot(x = "Weight", y = "Height", hue = "Index", data = data)


# In[37]:


sns.relplot(x = "Index", y = "Height", hue = "Gender", data = data, height = 5, aspect = 3)


# In[40]:


sns.relplot(x = "Index", y = "Height", hue = "Gender", kind = "line", data = data, height = 5, aspect = 3)


# In[42]:


x = data.drop('Index', axis=1)
y = data["Index"]


# In[45]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[46]:


s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.transform(x_test)


# In[48]:


clf = svm.SVC()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)


# In[50]:


print(classification_report(y_test, pred))


# In[52]:


print(confusion_matrix(y_test, pred))


# In[53]:


print(accuracy_score(y_test, pred))


# In[59]:


a = [[0,100,20]]
a = s.transform(a)
b = clf.predict(a)


# In[60]:


b


# In[56]:


data

