#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('C:\python\HR_comma_sep.csv')


# In[3]:


df.head()


# In[4]:


#数据清理，检查数据是否有缺失值
df.isnull().any()


# In[5]:


#适当改名
df = df.rename(columns={'satisfaction_level': 'satisfaction_level', 
                        'last_evaluation': 'last_evaluation',
                        'number_project': 'number_project',
                        'average_montly_hours': 'average_montly_hours',
                        'time_spend_company': 'time_spend_company',
                        'Work_accident': 'Work_accident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'left'
                        })


# In[6]:


#定类数据编码
df1=pd.Series(df['department']).unique()
df2=pd.Series(df['salary']).unique()
df['department'].replace(list(pd.Series(df['department']).unique()), np.arange(10), inplace = True)
df['salary'].replace(list(pd.Series(df['salary']).unique()), np.arange(3), inplace = True)


# In[7]:


#把left列移到表的前面，方便分析
front = df['left']
df.drop(labels = 'left', axis =1, inplace= True)
df.insert(0, 'left', front)


# In[8]:


df.head()


# In[9]:


#查看数据结构和类型
df.shape


# In[10]:


df.dtypes


# In[11]:


left_rate = df.left.value_counts()/14999
print(left_rate)


# In[12]:


#描述性分析
left_summary = df.groupby('left')

format = lambda x: '%.2f'%x
df.describe().applymap(format)


# In[13]:


#探索性分析
plt.subplots(figsize=(18,16)) 
sns.heatmap(df.corr(), annot = True)
plt.title('Heatmap of Correlation Matrix')


# In[14]:


#离职率与部门
left_1d = df['left'].groupby(df['department']).sum()
left_sum_d = df['left'].groupby(df['department']).count()
left_depart_rate = left_1d/left_sum_d
left_depart_rate.plot(kind = 'bar', figsize = (10, 8))
plt.title('Turnover rate of different departments')
print(left_depart_rate)


# In[15]:


#不同部门的平均薪资水平
salary_depart = df['salary'].groupby(df['department']).sum()
salary_mean = salary_depart/df['salary'].groupby(df['department']).count()
salary_mean.plot(kind = 'bar', figsize = (10, 8))
plt.title('Salary rate of different departments')


# In[16]:


#离职率与薪资
left_salary = df['left'].groupby(df['salary']).sum()
sum_salary = df['left'].groupby(df['salary']).count()
left_salary_rate = left_salary/sum_salary
left_salary_rate.plot(kind = 'bar', figsize = (10, 8))
plt.title('Turnover rate of different salary')


# In[17]:


#离职率与是否升职
left_promotion = df['left'].groupby(df['promotion']).sum()
sum_promotion = df['left'].groupby(df['promotion']).count()
left_promotion_rate = left_promotion/sum_promotion
left_promotion_rate.plot(kind = 'bar', figsize = (10, 8))
plt.title('Turnover rate of different promotion')


# In[18]:


#离职率与项目数量
left_number_project = df['left'].groupby(df['number_project']).sum()
sum_number_project = df['left'].groupby(df['number_project']).count()
left_number_project_rate = left_number_project/sum_number_project
left_number_project_rate.plot(kind = 'bar', figsize = (10, 8))
plt.title('Turnover rate of different project numbers')


# In[19]:


#离职率与入职公司时间
left_time_spend_company = df['left'].groupby(df['time_spend_company']).sum()
sum_time_spend_company = df['left'].groupby(df['time_spend_company']).count()
left_time_spend_company_rate = left_time_spend_company/sum_time_spend_company
left_time_spend_company_rate.plot(kind = 'bar', figsize = (10, 8))
plt.title('Turnover rate of different time_spend_company')


# In[20]:


#平均每月工作时间与离职率的关系
hours_left_table = pd.crosstab(index=df['average_montly_hours'], columns = df['left'])
fig = plt.figure(figsize = (10, 5))
letf = sns.kdeplot(df.loc[(df['left'] == 0), 'average_montly_hours'], color='b', shade = True, label='no left')
left = sns.kdeplot(df.loc[(df['left'] == 1), 'average_montly_hours'], color='r', shade = True, label='left')
plt.legend()


# In[21]:


#绩效评估与离职率的关系
evaluation_left_table=pd.crosstab(index=df['last_evaluation'],columns=df['left'])
fig=plt.figure(figsize=(10,5))
letf=sns.kdeplot(df.loc[(df['left']==0),'last_evaluation'],color='b',shade=True,label='no left')
left=sns.kdeplot(df.loc[(df['left']==1),'last_evaluation'],color='r',shade=True,label='left')
plt.legend()


# In[22]:


#满意度与离职率的关系
satisfaction_left_table=pd.crosstab(index=df['satisfaction_level'],columns=df['left'])
fig=plt.figure(figsize=(10,5))
letf=sns.kdeplot(df.loc[(df['left']==0),'satisfaction_level'],color='b',shade=True,label='no left')
left=sns.kdeplot(df.loc[(df['left']==1),'satisfaction_level'],color='r',shade=True,label='left')
plt.legend()


# In[23]:


#last_evaluation  vs  satisfaction_level
df1 = df[df['left']==1]
fig, ax = plt.subplots(figsize=(10,10))
pd.plotting.scatter_matrix(df1[['satisfaction_level','last_evaluation']], color = 'k', ax = ax)
plt.savefig('scatter.png',dpi=1000,bbox_inches='tight')#dpi细腻度，bbox_inches保存的图片旁边是否有留白
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




