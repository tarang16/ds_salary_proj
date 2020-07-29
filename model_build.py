#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("eda_data.csv")


# In[3]:


df.columns


# In[4]:


df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]


# In[5]:


# get dummy data 
df_dum = pd.get_dummies(df_model)


# In[6]:


# train test split 
from sklearn.model_selection import train_test_split


# In[7]:


X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


# multiple linear regression 
import statsmodels.api as sm


# In[10]:


X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


# In[11]:


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score


# In[12]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[13]:


np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))


# In[14]:


# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))


# In[15]:


alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))


# In[16]:


plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]


# In[17]:


# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[20]:


rf.fit(X_train,y_train)


# In[18]:


tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)


# In[21]:


tpred_rf = rf.predict(X_test)


# In[23]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)


# In[24]:


mean_absolute_error(y_test,tpred_lml)


# In[25]:


mean_absolute_error(y_test,tpred_rf)


# In[26]:


mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)


# In[ ]:




