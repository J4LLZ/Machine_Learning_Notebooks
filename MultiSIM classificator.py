#!/usr/bin/env python
# coding: utf-8

# ### Task Description 
# 
# The objective is to predict whether the customer is multisim (uses more than one SIM Card).
# 
# Given data is represented by the following features: 
# * **id**                           : unique customer id 
# * **mg_traffic_in**                : longdistance traffic 
# * **tp_code**                      : tariff plan code 
# * **num_voice_out**                : outgoing calls
# * **tp_change_date**               : tariff plan change date
# * **uniq_calls_cnt**               : number of unique phone numbers called
# * **age_cat**                      : age category 
# * **tech_sms_cnt_3m**              : number of technical SMS for 3 months
# * **tech_sms_cnt_6m**              : number of technical SMS for 6 months
# * **called_ctn_all_group**         : incoming calls group 
# * **device_cost**                  : device cost
# * **traf_kb**                      :  internet traffic (kb) 
# * **sim_type**                     : device sim type
# * **gender_male_prob**             : gender probability
# * **complex_value_sum**            : complex customer value 	
# * **complex_value_size**           : complex customer size 
# * **imei_cnt**                     : IMEI count  
# * **main_balance_adjust_minus_cnt**: number of balance adjustments  
# * **state_code**                   : region code
# * **macro_state**                  : macro region 
# * **time_onnet**                   : time on net  
# * **sum_recharges_3m**             : recharges sum for 3 months
# * **year**                    : year
# * **month**                        : month 
# * **target**                       : is multisim flag	
# 
# 
# 

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


# In[2]:


matplotlib.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 25)


# In[3]:


df = pd.read_excel(r'C:\Users\artyo\OneDrive\Рабочий стол\task\multisim_dataset.xlsx')

# Some rows have missing target value. Drop them
df = df[df['target'].isna() == False]

print('Data dimension:', df.shape)


# In[4]:


# Rows example 
df.head(7)


# In[5]:


# Constructing date using year and month

df['dt'] = pd.to_datetime(df['month'].apply(lambda x: str(x) if x >= 10 else '0' + str(x)) + '-' + df['year'].astype(str))
df.drop(columns = ['month', 'year'],
        inplace = True)


# In[6]:


# Number of unique customers

print('Number of unique customers:', df['id'].unique().shape[0])


# In[7]:


# Checking for duplicates for each pair id/dt 

df[['id', 'dt']].duplicated().sum()


# In[8]:


# Seems to be that each id has rows with different date 
df.groupby('id').size().describe()


# In[9]:


# Data filled unevenly for different date
# Feature 'tp_change_date' can be broadcasted for all id rows if 'tp_change_date' less that date 

df[df['id'] == 112]


# In[10]:


# Checking if there are customers with different target on different date
# No such ids 
df[['id', 'target']].drop_duplicates().groupby(['id', 'target']).size().describe()


# In[11]:


# Checking missing data rate for different dates 
date_list = df['dt'].unique()

for dt in date_list:
    print('DATE:',dt,
          end = '\n\n')
    
    print(df[df['dt']  == dt].isna().sum(),
          end = '\n\n')
    
    print('Total missing values:', df[df['dt']  == dt].isna().sum().sum(),
           end = '\n\n\n\n')


# #### Data preprocessing strategy
# 1. Considering the fact that customer's target is constant for each customer id, data should include **only one row per each id** to represent each customer as unique (deduplication).
# 2. Some features (such as tp_change_date) could be broadcasted for each customer id from one date to others.
# 3. The date representing each id should be choosen by the least missing data rate.
# 
# <br>
# In view of the above, choosen date is '2019-01-01' as the moment of least missing data (considering broadcasting from '2019-02-01')

# In[12]:


# broadcasting 'tp_change_date' and constructing flag of tariff plan changing 

df['tp_change_date'] = df.groupby('id')['tp_change_date'].apply(lambda x: x.bfill().ffill())
df['tp_change_flg'] = (df['dt'] > df['tp_change_date']).apply(lambda x: 1 if x else 0)
                                              
df.drop(columns = ['tp_change_date'],
        inplace = True)


# In[13]:


# moment of the least missing data 

df = df[df['dt'] == '2019-01-01']


# Iterating through different features shown that most valuable combination includes the following:

# In[14]:


# Flag of dualsim customer device 

df['is_dual'] = df['sim_type'].apply(lambda x: 1 if x == 'DualSim' else 0)
df.drop(columns = ['sim_type'], inplace = True)


# Missing values mean absence of balance adjustings/ technical sms 
df['tech_sms_cnt_3m'].fillna(0, inplace = True)
df['tech_sms_cnt_6m'].fillna(0, inplace = True)


# Missing rates of 'imei_cnt' and 'complex_value_sum' are close to zero 
df['imei_cnt'].fillna(df['imei_cnt'].mode().values[0], inplace = True)
df['complex_value_sum'].fillna(df['complex_value_sum'].mode().values[0], inplace = True )


# In[15]:


features = ['macro_state', 'mg_traffic_in', 'num_voice_out', 'tech_sms_cnt_3m', 'tech_sms_cnt_6m',
            'complex_value_sum', 'imei_cnt', 'gender_male_prob', 'time_onnet', 'is_dual', 'tp_change_flg']


# #### Model building

# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score


# In[17]:


train_df, test_df = train_test_split(df[features + ['target']],
                                     test_size =  0.3)

X_train, Y_train = train_df.drop(columns = 'target'), train_df['target']
X_test, Y_test = test_df.drop(columns = 'target'), test_df['target']


# In[29]:


# Best rf parameters
clf = RandomForestClassifier(max_depth = 5,
                             n_estimators = 500,
                             max_features = 4,
                             class_weight = 'balanced')

clf = clf.fit(X_train, Y_train)
Y_predicted = clf.predict(X_test)


# In[30]:


# Result metrics 

accuracy = accuracy_score(Y_predicted, Y_test)
recall = recall_score(Y_predicted, Y_test)
precision = precision_score(Y_predicted, Y_test)

print('Accuracy score:', accuracy,
      end = '\n\n')
print('Recall score:', recall,
      end = '\n\n')
print('Precision score:', precision)


# In[31]:


# feature ranking
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print('Feature ranking:')
for f in range(X_train.shape[1]):
    print('%s: %f' % (X_train.columns[indices[f]], importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
plt.title('Feature importances')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        color = 'green', 
        align = 'center')

plt.xticks(range(X_train.shape[1]), X_train.columns,  rotation = 30)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[ ]:




