#!/usr/bin/env python
# coding: utf-8

# This project has been created by trying to implement the K Nearest Neighbors Algorithm and Naïve Bayes Algorithm on two different datasets as a comparison and experimenting with different codes to predict heart disease using the patients symptoms. I initially thought i would use matlab however it is harder to use it for anything that isn’t simply represented as a numeric feature matrix and i wanted to explore both machine learning and deep learning on datasets hence decided to use Jupyter notebook to be able to do both. I have so far started developing some famous machine learning classifying algorithms. Next I will try to develope deep learning model to also do classification on my heart disease dataset and compare. There is still a lot to do but some basic functionalities created have allowed an insight to my research project and extended by knowledge to the side of classification since in my modules such as neural networks we are mainly using regression.

# In[79]:


import csv
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
 
# reading CSV using Pandas and storing in dataframe       
df = pd.read_csv('heartdisease.csv', header = None)

training_x=df.iloc[1:df.shape[0],0:13]
#set training set from 1-13

training_y=df.iloc[1:df.shape[0],13:14]
#set testing set the rest


# converting dataframe into arrays
x=np.array(training_x)
y=np.array(training_y)

for z in range(10):
    print("\n\n\nTest Train Split no. ",z+1,"\n\n\n")
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=None)
    # Gaussian function of sklearn
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())
    y_pred = gnb.predict(x_test)
    
    print("\n\nGaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    
    # convert 2D array to 1D array
    #y1=y_test.ravel()
    #y_pred1=y_pred.ravel()
    
    #print("\n\n\n\nConfusion Matrix")
    #cf_matrix=confusion_matrix(y1,y_pred1)
    #print(cf_matrix)
    
    # Matrix from 1D array
    #y2=np.zeros(shape=(len(y1),5))
    #y3=np.zeros(shape=(len(y_pred1),5))
    #for i in range(len(y1)):
       # y2[i][int(y1[i])]=1
    
   # for i in range(len(y_pred1)):
        #y3[i][int(y_pred1[i])]=1
        
    


# The result of the Naïve Bayes classifier on the heart disease data from UCI Machine Learning Repository is 55.2%

# In[2]:


import csv
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# reading CSV using Pandas and storing in dataframe       
df = pd.read_csv('heart_disease_dataset.csv', header = None)

training_x=df.iloc[1:df.shape[0],0:13]
#set training set from 1-13

training_y=df.iloc[1:df.shape[0],13:14]
#set testing set the rest


# converting dataframe into arrays
x=np.array(training_x)
y=np.array(training_y)

for z in range(10):
    print("\n\n\nTest Train Split no. ",z+1,"\n\n\n")
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=None)
    # Gaussian function of sklearn
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())
    y_pred = gnb.predict(x_test)
    
    print("\n\nGaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    
    # convert 2D array to 1D array
    #y1=y_test.ravel()
    #y_pred1=y_pred.ravel()
    
    #print("\n\n\n\nConfusion Matrix")
    #cf_matrix=confusion_matrix(y1,y_pred1)
    #print(cf_matrix)
    
    # Matrix from 1D array
    #y2=np.zeros(shape=(len(y1),5))
    #y3=np.zeros(shape=(len(y_pred1),5))
    #for i in range(len(y1)):
    #    y2[i][int(y1[i])]=1
    
    #for i in range(len(y_pred1)):
    #    y3[i][int(y_pred1[i])]=1
        
        


# The result of the Naïve Bayes classifier on the heart disease dataset from kaggles heart disease dataset is 85.5%

# In[76]:




from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('heartdisease2.csv')
df.info()
df.describe()

#dataset = pd.get_dummies(df,columns = ['sex' , 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
dataset = pd.get_dummies(df,columns = ['age','sex' , 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak' ])
#dataset = pd.get_dummies(df,columns = ['age','sex' , 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak', 'slope', 'ca','thal','num','target'])


# In[77]:


y=dataset['target']
x=dataset.drop(['target'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))
    
from sklearn.model_selection import cross_val_score
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,x,y,cv=10)
score.mean()


# The result of the K Neighbors classifier on the heart disease data from UCI Machine Learning Repository is 53.7%

# In[78]:


from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('heart_disease_dataset.csv')
df.info()
df.describe()

dataset = pd.get_dummies(df,columns = ['sex' , 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
#dataset = pd.get_dummies(df,columns = ['age','sex' , 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak' ])
#dataset = pd.get_dummies(df,columns = ['age','sex' , 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak', 'slope', 'ca','thal','num','target'])


# In[55]:


y=dataset['target']
x=dataset.drop(['target'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))
    
from sklearn.model_selection import cross_val_score
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,x,y,cv=10)
score.mean()


# The result of the K Neighbors classifier on the heart disease dataset from kaggles heart disease dataset is 66.5%

# So far according to the results the second dataset is performing a lot better in heart disease prediction using both algorithms
# as more patients with the symptoms as according to the dataset would suffer from heart diseases.
# 
# 
# K fold validation on chosen dataset

# In[66]:


import pandas as pd
mydata = pd.read_csv('heart_disease_dataset.csv')
mydata.head()


# In[67]:


import numpy as np

X = np.array(mydata[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']])
X.shape


# In[68]:


y = np.array(mydata['target'])
y.shape


# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size = 0.25)
X_test.shape


# In[70]:


from sklearn.linear_model import LinearRegression
model= LinearRegression()
mymodel =model.fit(X_train, y_train)


# In[71]:


# k fold
from sklearn.model_selection import cross_val_score
# 3 folds
scores = cross_val_score(mymodel,X_train, y_train, scoring='r2', cv =3)
scores


# In[72]:


# average score
print(np.mean(scores))


# In[73]:


# scores from test dataset
from sklearn.model_selection import cross_val_predict
prediction = cross_val_predict(model,X_test, y_test)
prediction


# In[74]:


# 3 fold cross validation on test dataset
scores_test = cross_val_score(model,X_test, y_test, cv =3)
scores_test


# In[75]:


#average 
print(np.mean(scores_test))


# 0.4418625710048419  with 3 folds so far

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




