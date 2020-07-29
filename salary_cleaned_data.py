#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
url = 'https://raw.githubusercontent.com/PlayingNumbers/ds_salary_proj/master/glassdoor_jobs.csv'
df = pd.read_csv(url, error_bad_lines=False)


# In[3]:


df.head()


# In[5]:


df.to_csv("glassdoor.csv")


# In[6]:


df=pd.read_csv("glassdoor.csv")


# In[7]:


df.head()


# In[10]:


df["Salary Estimate"]


# In[11]:


df=df[df["Salary Estimate"]!='-1']


# In[12]:


df.info()


# In[15]:


salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])


# In[16]:


salary


# In[56]:


minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))


# In[57]:


min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))


# In[22]:


df['hourly']=df["Salary Estimate"].apply(lambda x: 1 if 'per hour' in x.lower() else 0)


# In[58]:


df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))


# In[59]:


df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))


# In[60]:


df['avg_salary'] = (df.min_salary+df.max_salary)/2


# In[61]:


df['employer_provided']=df["Salary Estimate"].apply(lambda x: 1 if'employer provided salary:' in x.lower() else 0)


# In[62]:


df['employer_provided']


# In[64]:


df.head()


# In[32]:


df


# In[55]:


df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))


# In[53]:


df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))


# In[51]:


df['min_salary']


# In[42]:


df.info()


# In[65]:


df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)


# In[66]:


df['company_txt']


# In[67]:


df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df.job_state.value_counts()


# In[68]:


df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)


# In[69]:


df['age'] = df.Founded.apply(lambda x: x if x <1 else 2020 - x)


# In[70]:


df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)


# In[71]:


df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df.R_yn.value_counts()


# In[72]:


df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark.value_counts()


# In[73]:


df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws.value_counts()


# In[74]:


df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel.value_counts()


# In[75]:


df_out = df.drop(['Unnamed: 0'], axis =1)


# In[76]:


df_out.to_csv('salary_data_cleaned.csv',index = False)


# In[ ]:




