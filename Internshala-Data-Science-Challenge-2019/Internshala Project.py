#!/usr/bin/env python
# coding: utf-8

# # Hypothesis Generation

# Below are some of the factors which I think can affect the target (dependent variable for this premium-paying prediction problem):
#     1. Applicants with highest precentage of paying a premium should have more chances to pay in time.
#     2. Applicants who's age is not too old may have more probability to pay in time.
#     3. Applicants with high income should have more chances of paying a premium.
#     4. Applicants who resides in Urban, higher chances of paying a premium.
#     5. Applicants with pay premium on time several times, higher chances.

# # Loading of data and importing the libraries and packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")


# In[2]:


from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#     Making a copy of train and test data 

# In[4]:


train_original = train.copy()
test_original = test.copy()


# # Understanding the data

# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


test.shape, train.shape


#     Dropping premium column

# In[8]:


train = train.drop('premium', 1)
train_original = train_original.drop('premium', 1)


# In[9]:


train.dtypes


# In[10]:


train.describe()


# In[11]:


train.isnull().sum()


# # Univariate Analysis

#     Continuous Data Object

# # no_of_premiums_paid column

# In[12]:


train['no_of_premiums_paid'].describe()


#     Detecting outliers using IQR Method

# In[13]:


Q1_premiumpaid_train = train['no_of_premiums_paid'].quantile(0.25)
Q3_premiumpaid_train = train['no_of_premiums_paid'].quantile(0.75)
IQR_premiumpaid_train = Q3_premiumpaid_train  - Q1_premiumpaid_train 
upper_premiumpaid_train = Q3_premiumpaid_train + 1.5 * IQR_premiumpaid_train
lower_premiumpaid_train = Q1_premiumpaid_train - 1.5 * IQR_premiumpaid_train
upper_premiumpaid_train, lower_premiumpaid_train


# In[14]:


Q1_premiumpaid_test = test['no_of_premiums_paid'].quantile(0.25)
Q3_premiumpaid_test = test['no_of_premiums_paid'].quantile(0.75)
IQR_premiumpaid_test = Q3_premiumpaid_test - Q1_premiumpaid_test 
upper_premiumpaid_test = Q3_premiumpaid_test + 1.5 * IQR_premiumpaid_test
lower_premiumpaid_test = Q1_premiumpaid_test - 1.5 * IQR_premiumpaid_test
upper_premiumpaid_test, lower_premiumpaid_test


# In[15]:


print(train[train['no_of_premiums_paid'] < -3.5]['no_of_premiums_paid'].count())
print(train[train['no_of_premiums_paid'] > 24.5]['no_of_premiums_paid'].count())
print(test[test['no_of_premiums_paid'] < -3.5]['no_of_premiums_paid'].count())
print(test[test['no_of_premiums_paid'] > 24.5]['no_of_premiums_paid'].count())


# In[16]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['no_of_premiums_paid'].plot.box(ax=axarr[0])
train['no_of_premiums_paid'].plot.box(ax=axarr[1])


# In[17]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['no_of_premiums_paid'].hist(ax=axarr[0])
train['no_of_premiums_paid'].hist(ax=axarr[1])


# train.loc[train['no_of_premiums_paid'] > upper_premiumpaid_train ,'no_of_premiums_paid'] = np.mean(train['no_of_premiums_paid'])
# train.loc[train['no_of_premiums_paid'] < lower_premiumpaid_train ,'no_of_premiums_paid'] = np.mean(train['no_of_premiums_paid'])
# train['no_of_premiums_paid'].plot.box()

# test.loc[test['no_of_premiums_paid'] > upper_premiumpaid_test,'no_of_premiums_paid'] = np.mean(test['no_of_premiums_paid'])
# test.loc[test['no_of_premiums_paid'] < lower_premiumpaid_test,'no_of_premiums_paid'] = np.mean(test['no_of_premiums_paid'])
# test['no_of_premiums_paid'].plot.box()

# # application_underwriting_score column

#     Checking whether application_underwriting_score lies below 90

# In[18]:


test[test['application_underwriting_score']<90].shape[0] == 0,train[train['application_underwriting_score']<90].shape[0] == 0


# In[19]:


train['application_underwriting_score'].mode()[0]


# In[20]:


train['application_underwriting_score'].describe()


#     Detecting outliers using IQR Method

# In[21]:


Q1_application_train = train['application_underwriting_score'].quantile(0.25)
Q3_application_train = train['application_underwriting_score'].quantile(0.75)
IQR_application_train = Q3_application_train  - Q1_application_train 
upper_application_train = Q3_application_train + 1.5 * IQR_application_train
lower_application_train = Q1_application_train - 1.5 * IQR_application_train
upper_application_train, lower_application_train


# In[22]:


Q1_application_test = test['application_underwriting_score'].quantile(0.25)
Q3_application_test = test['application_underwriting_score'].quantile(0.75)
IQR_application_test = Q3_application_test - Q1_application_test 
upper_application_test = Q3_application_test + 1.5 * IQR_application_test
lower_application_test = Q1_application_test - 1.5 * IQR_application_test
upper_application_test, lower_application_test


# In[23]:


print(train[train['application_underwriting_score'] < lower_application_train]['application_underwriting_score'].count())
print(train[train['application_underwriting_score'] > upper_application_train]['application_underwriting_score'].count())
print(test[test['application_underwriting_score'] < lower_application_test]['application_underwriting_score'].count())
print(test[test['application_underwriting_score'] > upper_application_test]['application_underwriting_score'].count())


# In[24]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['application_underwriting_score'].plot.box(ax=axarr[0])
train['application_underwriting_score'].plot.box(ax=axarr[1])


# In[25]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['application_underwriting_score'].hist(ax=axarr[0])
train['application_underwriting_score'].hist(ax=axarr[1])


# train.loc[train['application_underwriting_score'] > upper_application_train,'application_underwriting_score'] = np.mean(train['application_underwriting_score'])
# train.loc[train['application_underwriting_score'] < lower_application_train,'application_underwriting_score'] = np.mean(train['application_underwriting_score'])
# train['application_underwriting_score'].plot.box()

# test.loc[test['application_underwriting_score'] > upper_application_test,'application_underwriting_score'] = np.mean(test['application_underwriting_score'])
# test.loc[test['application_underwriting_score'] < lower_application_test,'application_underwriting_score'] = np.mean(test['application_underwriting_score'])
# test['application_underwriting_score'].plot.box()

# # Income column

# In[26]:


train['Income'].describe()


#     Detecting outliers using IQR Method

# In[27]:


Q1_Income_train = train['Income'].quantile(0.25)
Q3_Income_train = train['Income'].quantile(0.75)
IQR_Income_train = Q3_Income_train  - Q1_Income_train 
upper_Income_train = Q3_Income_train + 1.5 * IQR_Income_train
lower_Income_train = Q1_Income_train - 1.5 * IQR_Income_train
upper_Income_train, lower_Income_train


# In[28]:


Q1_Income_test = test['Income'].quantile(0.25)
Q3_Income_test = test['Income'].quantile(0.75)
IQR_Income_test = Q3_Income_test - Q1_Income_test 
upper_Income_test = Q3_Income_test + 1.5 * IQR_Income_test
lower_Income_test = Q1_Income_test - 1.5 * IQR_Income_test
upper_Income_test, lower_Income_test


# In[29]:


print(train[train['Income'] < lower_Income_train]['Income'].count())
print(train[train['Income'] > upper_Income_train]['Income'].count())
print(test[test['Income'] < lower_Income_test]['Income'].count())
print(test[test['Income'] > upper_Income_test]['Income'].count())


# In[30]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['Income'].plot.box(ax=axarr[0],showfliers=False)
train['Income'].plot.box(ax=axarr[1],showfliers=False)


# In[31]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['Income'].plot.hist(ax=axarr[0])
train['Income'].plot.hist(ax=axarr[1])


# In[32]:


train['Income'].min(), train['Income'].max()


# train.loc[train['Income'] > upper_Income_train,'Income'] = np.mean(train['Income'])
# train.loc[train['Income'] < lower_Income_train,'Income'] = np.mean(train['Income'])
# train['Income'].plot.box()

# test.loc[test['Income'] > upper_Income_test,'Income'] = np.mean(test['Income'])
# test.loc[test['Income'] < lower_Income_test,'Income'] = np.mean(test['Income'])
# test['Income'].plot.box()

# # age_in_days column

# In[33]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['age_in_days'].hist(ax=axarr[0])
train['age_in_days'].hist(ax=axarr[1])


# In[34]:


test['age_in_days'] = (test['age_in_days']/365).astype(int)
train['age_in_days'] = (train['age_in_days']/365).astype(int)


#     Renaming a column as age in days converted into years

# In[35]:


train = train.rename(columns={"age_in_days": "age_in_years"})
test = test.rename(columns={"age_in_days": "age_in_years"})


# In[36]:


train.columns


#     Detecting outliers using IQR Method

# In[37]:


Q1_age_train = train['age_in_years'].quantile(0.25)
Q3_age_train = train['age_in_years'].quantile(0.75)
IQR_age_train = Q3_age_train  - Q1_age_train 
upper_age_train = Q3_age_train + 1.5 * IQR_age_train
lower_age_train = Q1_age_train - 1.5 * IQR_age_train
upper_age_train, lower_age_train


# In[38]:


Q1_age_test = test['age_in_years'].quantile(0.25)
Q3_age_test = test['age_in_years'].quantile(0.75)
IQR_age_test = Q3_age_test - Q1_age_test 
upper_age_test = Q3_age_test + 1.5 * IQR_age_test
lower_age_test = Q1_age_test - 1.5 * IQR_age_test
upper_age_test, lower_age_test


# In[39]:


print(train[train['age_in_years'] < lower_age_train]['age_in_years'].count())
print(train[train['age_in_years'] > upper_age_train]['age_in_years'].count())
print(test[test['age_in_years'] < lower_age_test]['age_in_years'].count())
print(test[test['age_in_years'] > upper_age_test]['age_in_years'].count())


# In[40]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['age_in_years'].plot.box(ax=axarr[0])
train['age_in_years'].plot.box(ax=axarr[1])


# In[41]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['age_in_years'].hist(ax=axarr[0])
train['age_in_years'].hist(ax=axarr[1])


# train.loc[train['age_in_days'] > upper_age_train,'age_in_days'] = np.mean(train['age_in_days'])
# train.loc[train['age_in_days'] < lower_age_train,'age_in_days'] = np.mean(train['age_in_days'])
# train['age_in_days'].plot.box()

# test.loc[test['age_in_days'] > upper_age_test,'age_in_days'] = np.mean(test['age_in_days'])
# test.loc[test['age_in_days'] < lower_age_test,'age_in_days'] = np.mean(test['age_in_days'])
# test['age_in_days'].plot.box()

# # perc_premium_paid_by_cash_credit column

# In[42]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['perc_premium_paid_by_cash_credit'].plot.box(ax=axarr[0])
train['perc_premium_paid_by_cash_credit'].plot.box(ax=axarr[1])


# In[43]:


fig, axarr = plt.subplots(1,2, figsize=(15, 5))
test['perc_premium_paid_by_cash_credit'].hist(ax=axarr[0])
train['perc_premium_paid_by_cash_credit'].hist(ax=axarr[1])


# # Count_3-6_months_late column

# # Count_6-12_months_late column

# # Count_more_than_12_months_late column

# In[44]:


plt.hist(train['Count_3-6_months_late'], bins=30, label="Count_3-6_months_late")
plt.hist(train['Count_6-12_months_late'], bins=30, label="Count_6-12_months_late")
plt.hist(train['Count_more_than_12_months_late'], bins=30, label="Count_more_than_12_months_late")
plt.legend()
plt.show()


# In[45]:


train['Count_more_than_12_months_late'].value_counts()


# In[46]:


train['Count_6-12_months_late'].value_counts()


# In[47]:


train['Count_3-6_months_late'].value_counts()


#     Categorical Data Object

# # sourcing_channel column

# In[48]:


train['sourcing_channel'].value_counts()/len(train['sourcing_channel']) * 100


# In[49]:


test['sourcing_channel'].value_counts()/len(test['sourcing_channel']) * 100


# In[50]:


train['sourcing_channel'].value_counts(normalize=True).plot.bar(title = 'Sourcing Channel')


# # residence_area_type column

# In[51]:


train['residence_area_type'].value_counts(normalize=True)


# In[52]:


test['residence_area_type'].value_counts(normalize=True)


# In[53]:


train['residence_area_type'].value_counts(normalize=True).plot.bar(title = 'Residence Area')


# # Mapping Categorical Values

# In[54]:


train["sourcing_channel"] = train["sourcing_channel"].map({'A':1,'B':2,'C':3,'D':4,'E':5})
train["residence_area_type"] = train["residence_area_type"].map({'Rural':1,'Urban':2})


# In[55]:


test["sourcing_channel"] = test["sourcing_channel"].map({'A':1,'B':2,'C':3,'D':4,'E':5})
test["residence_area_type"] = test["residence_area_type"].map({'Rural':1,'Urban':2})


# # Bivariate Analysis

# In[56]:


train.columns


# In[57]:


Sourcing_Channel = pd.crosstab(train['sourcing_channel'], train['target'])
Sourcing_Channel.div(Sourcing_Channel.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(5,5))
plt.show()


# In[58]:


Residence = pd.crosstab(train['residence_area_type'], train['target'])
Residence.div(Residence.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(5,5))
plt.show()
Residence


# In[59]:


train['Income'].min(), train['Income'].max(), train['Income'].head()


# In[60]:


bins = [24000, 100000, 150000,500000, 90262600] 
group = ['Low','Average','High', 'Very high'] 
train['Income_bin'] = pd.cut(train['Income'],bins,labels=group)
Income_bin = pd.crosstab(train['Income_bin'],train['target'])

Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar") 
plt.xlabel('Income') 
P = plt.ylabel('Percentage')


# In[61]:


del train['Income_bin']


# In[62]:


plt.figure(figsize=(14,12))
sns.heatmap(train.drop('id',axis=1).corr(), annot=True)


# In[63]:


train[["target","residence_area_type"]].groupby(["residence_area_type"]).mean()


# In[64]:


train[["target","sourcing_channel"]].groupby(["sourcing_channel"]).mean()


# # Missing Impuatation and Outliers Treatment

# In[65]:


train.isnull().sum()


#     Income Column Normalization for bigger numbers

# In[66]:


train['Income'] = train['Income'].apply(np.log).round(2)
test['Income'] = test['Income'].apply(np.log).round(2)


# In[67]:


train['application_underwriting_score'].fillna(train['application_underwriting_score'].mode()[0],inplace=True)
test['application_underwriting_score'].fillna(train['application_underwriting_score'].mode()[0],inplace=True)


# In[68]:


#train.fillna(0,inplace=True)
#test.fillna(0,inplace=True)


# # Standard Scaler Preprocessing

# In[69]:


train.columns


# In[70]:


cols_for_ss = ['perc_premium_paid_by_cash_credit', 'age_in_years', 'Income',
       'Count_3-6_months_late', 'Count_6-12_months_late',
       'Count_more_than_12_months_late', 'application_underwriting_score',
       'no_of_premiums_paid', 'sourcing_channel', 'residence_area_type']

scaler = preprocessing.StandardScaler().fit(train[cols_for_ss])
train[cols_for_ss] = scaler.transform(train[cols_for_ss])
test[cols_for_ss] = scaler.transform(test[cols_for_ss])
print(scaler.mean_)


# In[71]:


train.head()


# # Build Training and Testing Model

# In[72]:


X = train.drop(['id','target'],axis=1)
y = train_original.target


# In[73]:


x_test = test.drop('id',axis=1)


# In[74]:


x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.95, test_size=0.05,shuffle=False)


# In[75]:


x_test.shape, x_train.shape,test.shape, train.shape


# In[76]:


gbr = XGBClassifier(missing=np.nan, 
                    learning_rate = 0.15, 
                    gamma=1, 
                    colsample_bytree=0.8)


# In[77]:


gbr.fit(X, y)


# In[78]:


pred_cv = gbr.predict(x_valid)


# In[79]:


roc_auc_score(y_valid, pred_cv)


# In[80]:


predict_target_proba = gbr.predict_proba(x_test)
predict_target = gbr.predict(x_test)
print(gbr.score(x_test, predict_target),gbr.score(x_train, y_train), gbr.score(x_valid,y_valid))

csv = pd.DataFrame()
csv['id'] = test['id']
csv['target'] = predict_target_proba[:,1]
csv.to_csv('sample.csv', header=True, index=False)


# In[81]:


predict_target_proba[:,1]

