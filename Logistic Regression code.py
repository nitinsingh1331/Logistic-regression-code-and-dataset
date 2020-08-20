#!/usr/bin/env python
# coding: utf-8

# # logistic regression 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
df=pd.read_csv("C:/Users/Dell/Downloads/Social_Network_Ads.csv")
df.head(30)


# In[2]:


x=df.iloc[:,:-1].values  
y=df.iloc[:,-1].values 


# In[3]:


from sklearn.model_selection import train_test_split  
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)  
print(X_train)
print(y_train)
print(X_test)
print(y_test)


# In[4]:





from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)











X_test = sc.transform(X_test)
print(X_train)
print(X_test)


# In[5]:


# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[6]:


# 30,87000 are in the original scale before applying feature scaling 
print(classifier.predict(sc.transform([[30,87000]])))


# In[7]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 


# In[8]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred) 
print(cm)
accuracy_score(y_test,y_pred)


# In[18]:


from matplotlib.colors import ListedColormap 
X_set,y_set=X_train,y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('k', 'white')))
plt.xlim(X1.min(),X1.max()) 
plt.ylim(X2.min(),X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("trining test")
plt.xlabel("age")
plt.ylabel("estimated salary")  
plt.legend()


# In[19]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('k', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 

