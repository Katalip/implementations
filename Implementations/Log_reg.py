#!/usr/bin/env python
# coding: utf-8

# In[156]:


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# In[183]:


df = pd.read_csv('Titanic.csv')


# In[185]:


df.head()


# In[186]:


df.info()


# In[187]:


for i in df.columns[1:]:
    print(f'Column: {i}, unique_values: {df[i].unique()}  Na_values: {df[i].isna().sum()}', end=" \n\n")


# In[188]:


plt.rcParams['figure.figsize'] = (15,8)


# ### Gender

# In[189]:


sns.countplot(df['Survived'], hue=df['Sex'], palette=['grey', next(color_cycle)])


# - **So we can map female to 1 and male to 0 (More men have dided afterall)**

# In[190]:


df['Sex'] = df['Sex'].map({'female':1, 'male':0})


# ### Age

# In[191]:


# Imputing NaN with mean
median_age = df['Age'].dropna().mean()
df['Age'].fillna(median_age, inplace=True)


# In[192]:


min_age = df['Age'].min()
max_age = df['Age'].max()
(min_age, max_age) 


# In[193]:


df.iloc[np.argmin(df['Age'])]


# In[194]:


age_groups = [0,10,20,30,40,50,60,75]
lbls = ['ag_' + str(age_groups[i]) + '_' + str(age_groups[i+1]) for i in range(len(age_groups)-1)]
df['Age_groups'] = pd.cut(df['Age'], bins = age_groups, labels = lbls)
df = pd.concat([df, pd.get_dummies(df['Age_groups'])], axis=1)


# In[168]:


df.tail()


# In[195]:


fig, axs = plt.subplots(2,4,figsize=(20,15))
axs = axs.flatten()
ax_idx = 0
for i, v in df.groupby('Age_groups'):
    sns.countplot(v['Survived'], ax=axs[ax_idx], palette=['#D3D3D3', next(color_cycle)])
    axs[ax_idx].set_title(i)
    ax_idx += 1
    


# In[196]:


df.drop('Age_groups', axis=1, inplace=True)


# ## Pclass

# In[197]:


sns.countplot(df['Survived'], hue=df['PClass'], color=next(color_cycle))


# In[198]:


df['PClass'] = df['PClass'].map({'1st':2, '2nd':1, '3rd':0})


# In[245]:


df['PClass'] = df['PClass'].fillna(0)


# ### Names

# In[199]:


df['Name'].apply(lambda x: x.split(' ')[1]).value_counts()


# In[200]:


titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Ms', 'Rev', 'Dr']
for i in titles:
    df['is_' + str(i)] = df['Name'].apply(lambda x: x.split(' ')[1] == i)
    df['is_' + str(i)] = df['is_' + str(i)].astype('int64') 
df.drop('Name',axis=1,inplace=True)


# In[201]:


df.head()


# ### Splitting for train_test

# In[176]:


# size = 0.7
# y = df['Survived']
# x = df.drop('Survived', axis=1)
# train_x, train_y = x[:int(len(x)*size)], y[:int(len(x)*size)]
# test_x, test_y = x[int(len(x)*size):], y[int(len(x)*size):]


# In[177]:


from sklearn.model_selection import train_test_split


# In[246]:


y = df['Survived']
x = df.drop('Survived', axis=1)


# In[247]:


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3)


# In[248]:


train_x.shape, train_y.shape


# ### Logistic Regression

# In[453]:


from tqdm import tqdm_notebook, tnrange 


# In[ ]:


def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[532]:


class SGDclass():
    def __init__(self, lr = 1e-2, n_epochs = 3000, random_state=17):
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.costs = []
        self.weights = []
        
    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        y = y[:, None]
        w = np.zeros((x.shape[1], 1))
        b = 0
        for i in tnrange(self.n_epochs, desc='Epochs'):
            m = x.shape[0]
            a = sigmoid(x@w + b)
            cost = -1/m * np.sum(y  * np.log(a) + (1-y)*np.log(1-a))
            dw = 1/m * (a-y).T@x
            db = 1/m * np.sum(a-y)
            #self.lr = 1/(i+1)
            w = w-self.lr*dw.T
            b = b-self.lr*db
            if(i%100 == 0):
                print(f'Epoch: {i} || Cost {cost}')
            self.costs.append(cost)
            self.weights.append([w,b])
            
    def predict(self, x):
        w = self.weights[np.argmin(self.costs)][0]
        b = self.weights[np.argmin(self.costs)][1]
        return (sigmoid(x@w + b) > 0.5)*1


# In[538]:


s = SGDclass()
s.fit(train_x, train_y)


# In[539]:


accuracy_score(test_y,s.predict(test_x))


# In[541]:


len(s.costs[::100])


# In[542]:


sns.lineplot(np.arange(0,3000,100),s.costs[::100])
plt.xlabel('Epochs')
plt.ylabel('Cost')


# ### Comparing with sklearns's sgdclass

# In[446]:


from sklearn.linear_model import SGDClassifier


# In[519]:


s = SGDClassifier()
s.fit(train_x, train_y)


# In[544]:


from sklearn.metrics import accuracy_score


# In[520]:


accuracy_score(test_y,s.predict(test_x))


# In[297]:


from sklearn.ensemble import RandomForestClassifier


# In[312]:


rf = RandomForestClassifier()


# In[313]:


rf.fit(train_x, train_y)


# In[314]:


accuracy_score(train_y,rf.predict(train_x))


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Log_reg.ipynb')

