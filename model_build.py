#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df=pd.read_csv("eda_data.csv")


# In[9]:


df.columns


# In[10]:


df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]


# In[11]:


# get dummy data 
df_dum = pd.get_dummies(df_model)


# In[12]:


# train test split 
from sklearn.model_selection import train_test_split


# In[13]:


X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


# multiple linear regression 
import statsmodels.api as sm


# In[16]:


X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


# In[17]:


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score


# In[18]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[19]:


np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))


# In[20]:


# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))


# In[21]:


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


# In[22]:


# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[23]:


rf.fit(X_train,y_train)


# In[24]:


tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)


# In[25]:


tpred_rf = rf.predict(X_test)


# In[26]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)


# In[27]:


mean_absolute_error(y_test,tpred_lml)


# In[28]:


mean_absolute_error(y_test,tpred_rf)


# In[29]:


mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)


# In[38]:


params={"learning_rate":[0.05,0.10,0.15,0.20,0.25,0.30],"max_depth":[3,4,5,6,8,10,12,15],"min_child_weight":[1,3,5,7],"gamma":[0.0,0.1,0.2,0.3,0.4],"colsample_bytree":[0.3,0.4,0.5,0.7]}


# In[46]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)


# In[42]:


from sklearn.model_selection import RandomizedSearchCV

classifier=xgboost.XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[48]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[49]:


rf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model

rf.fit(X_train,y_train)


# In[ ]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)


# In[51]:


rf.best_estimator_


# In[52]:


import pickle
pickl={'model': rf.best_estimator_}
pickle.dump(pickl,open('model_file'+".p","wb"))


# In[53]:


file_name="model_file.p"
with open (file_name,'rb') as pickled:
    data=pickle.load(pickled)
    model=data['model']


# In[56]:


model.predict(X_test.iloc[1,:].values.reshape(1,-1))


# In[57]:


list(X_test.iloc[1,:])


# In[ ]:




