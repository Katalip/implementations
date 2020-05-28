#!/usr/bin/env python
# coding: utf-8
# Variant 2
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# Reading our data
df = pd.read_csv('data/Iris.csv', index_col = 'Id')


# Cehcking first few rows
df.head()


# checking columns types
df.info()


# Checking mean,std values
# Values seem to be adequate, that's why we may not use Normalisation, Standardization
df.describe()

# Checking for missing values
df.isna().sum()


# Categorifying target variable
df['Species'] = df['Species'].astype('category')


# Checking column types again
df.info()


# Checking categorified target variable codes
df['Species'].cat.codes.unique()


# Specifying x and y
x, y = df.drop('Species', axis=1).values, df['Species']


## KMeans
def Kmeans(x, k=2, random_state=0, n_iters = 200):
    '''
    KMeans is a clustering algorihtm based on finding centroids which separate our instances in the best possible way.
    Algorithm: 1) Randomly pick k instance from x as centroids
               2) Compute distances from each observation to each centroid
               3) Assign nearest centroid's class to an observation
               4) Move centroid by calculating mean of grouped observations
               5) Repeat again till the convergence
    
    Argumetns:
    x - observations
    k - number of clusters
    random_state - specify to repeat the same results
    n_iters - number of iterations 
    '''
    np.random.seed(random_state) # Specifying random_state
    centroids = x[np.random.randint(0,x.shape[0], k), :] # Picking random centroids
    for i in range(n_iters): 
        c = [] # to store classes based on nearest centroid
        for i in x:
            dists = np.sqrt(np.sum((centroids - i)**2, axis=1)) # Computing distances (In this case i'm using Euclidian distances)
            c.append(np.argmin(dists))
        c = np.array(c) # converitng to numpy array to use boolean mapping
        for i in range(len(centroids)): 
            centroids[i] = np.sum(x[c == i], axis=0)/len(x[c == i]) # Moving centroids

    for i in range(len(centroids)):
        cost = np.sum(np.sum((x[c == i] - centroids[i])** 2, axis=1)) # Computing  sum of squared distances 
                                                                      # of each data point to it's assigned cluster

    return centroids, c, np.sum(cost)


costs = []
for k in range(1,11):
    centrs, classes, cost = Kmeans(x, k)
    costs.append(cost)


# We can use 'elbow' method to choose appropriate clusters number
# Here our line stops decreaing dramatically after k = 2, so we can choose k = 2

sns.lineplot(np.arange(1,11), costs) # 1.png 


c = classes
# Plotting our clusters and centroids
# Values for picked columns were chosen after tyring different random combinations
plt.scatter(x[c == 0, 0], x[c == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[c == 1, 0], x[c == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[c == 2, 0], x[c == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
sns.scatterplot(centroids[:, 0], centroids[:, 1], s = 100, color='yellow') # 2.png


## Comparing it with sklearn's Kmeans

# K-means from sklearn
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # within cluster sum of squares
plt.show() # 3.png

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init=10, random_state =0)
y_kmeans = kmeans.fit_predict(x)


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend() # 4.png


## Teting with another data

# To test our algorithm further we may 
# try to apply it on a dataset without explicit target variable
# Let's try it on TripAdvisor data

df2 = pd.read_csv('data/tripadvisor_review.csv')
df2.info()
df2.isna().sum()
df2.describe()

# It's seem that values don't need to be scaled

x = df2.drop('User ID', axis=1).values

# In order to plot clusters correctly I'm using PCA
# Before i tried without it, but didn't inlude that part
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
new_x = pca.fit_transform(x)

print(pca.explained_variance_)

costs = []
for k in range(1,11):
    centrs, classes, cost = Kmeans(new_x, k)
    costs.append(cost)


sns.lineplot(np.arange(1,11), costs) #5.png

# We may try 2 or 3 clusters

centrs, classes, cost = Kmeans(new_x, 3)
print(len(x[classes==0]), len(x[classes==1]), len(x[classes==2])) 

c = classes
# We may notice that instances were differentiated pretty well
sns.scatterplot(new_x[c == 0, 0], new_x[c == 0, 1], s = 100, color = 'red',)
sns.scatterplot(new_x[c == 1, 0], new_x[c == 1, 1], s = 100, color = 'blue')
sns.scatterplot(new_x[c == 2, 0], new_x[c == 2, 1], s = 100, color = 'green')
sns.scatterplot(centrs[:, 0], centrs[:, 1], s = 100, color='yellow')  #6.png

# Based on this we may specify a category for each user

# get_ipython().system('python notebook2script.py Kmeans.ipynb')

