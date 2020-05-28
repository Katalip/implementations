#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm_notebook, tnrange


# In[2]:


# reading data
df = pd.read_csv('kc_house_data.csv')


# ### Usually, one should do EDA at first, but assuming that this is a toy dataset I'm skipping it

# In[3]:


df.T


# In[4]:


plt.figure(figsize=(20,10))
sns.heatmap(df.iloc[:, 2:].corr(), annot=True)


# In[5]:


x,y = df.drop(['id','date','price'], axis=1), df['price']


# In[6]:


sns.distplot(y)


# In[7]:


# Applying log transformation
y = np.log(y)


# In[8]:


sns.distplot(y)


# In[9]:


# Spliting into train/test sets
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, shuffle=False)


# In[10]:


# Scaling data
ss = StandardScaler()
train_x = ss.fit_transform(train_x) # We should call fit_transform only on train
test_x = ss.transform(test_x)


# In[22]:


def mse(y,y_pred):
    return np.sum((y - y_pred)**2)/(len(y)) 


# In[88]:


from tqdm import tqdm_notebook, tnrange


# In[89]:


class SGDreg():
    def __init__(self, lr = 1e-3, n_epochs = 5, random_state=17):
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.mse_hist = []
        self.weights = []
        self.biases = []
    
    def fit(self, x, y):
        x = np.hstack((np.ones((x.shape[0],1)), x))
        w = np.zeros((x.shape[1]))
        for n in tnrange(self.n_epochs, desc='Epochs'):
            for i in tqdm_notebook(range(x.shape[0]), desc='SGD inner loop'):
                w = w + x[i]*self.lr*(2*(y[i] - np.dot(x[i],w)))
                
                mse = mean_squared_error(y,np.matmul(x,w.T))
                self.mse_hist.append(mse)
                self.weights.append(w)
        self.w = self.weights[np.argmin(self.mse_hist)]
            
    def predict(self, x):
        x = np.hstack((np.ones((x.shape[0],1)), x))
        return x@self.w


# In[90]:


s = SGDreg()


# In[86]:


get_ipython().run_line_magic('pinfo2', 'trange')


# In[91]:


s.fit(train_x, train_y)


# In[56]:


import gc
gc.collect()


# In[61]:


s.mse_hist


# In[63]:


s.predict(test_x).shape


# In[64]:


test_y.shape


# In[62]:


mse(s.predict(test_x), test_y)


# In[20]:


from sklearn.metrics import mean_squared_error


# In[67]:


mean_squared_error(s.predict(test_x), test_y[:, None])


# In[48]:


get_ipython().system('jupyter nbconvert --to script Lin_reg.ipynb')


# In[391]:


s.predict(test_x)

