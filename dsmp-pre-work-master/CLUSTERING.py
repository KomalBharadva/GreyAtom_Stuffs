# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# Load Offers
offers=pd.read_excel(path,sheet_name=0)

# Load Transactions
transactions=pd.read_excel(path,sheet_name=1)
transactions['n']=transactions.loc[:,'n']='1'

# Merge dataframes
df=pd.concat([offers,transactions])
# df.drop([df.index[0],df.index[33]],inplace=True)
# df=df.drop(df.index[0,33])
# df=df.drop([:33])
# df.reset_index()
df.drop(df.index[:16], inplace=True)

# Look at the first 5 rows
# print(df.head())
print(df.shape)




# --------------
# Code starts here

# create pivot table
matrix=df.pivot_table(index='Customer Last Name', columns='Offer #', values='n')

# replace missing values with 0
matrix.fillna(0, inplace=True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
print(matrix.head())

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster = KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10, random_state=0)

# create 'cluster' column
matrix['cluster']=cluster.fit_predict(matrix[matrix.columns[1:]])

print(matrix.head())
# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca=PCA(n_components=2, random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x']=pca.fit_transform(matrix[matrix.columns[1:]])[:,0]

matrix['y']=pca.fit_transform(matrix[matrix.columns[1:]])[:,1]

# dataframe to visualize clusters by customer names
# clusters=matrix.iloc([0, 33, 34, 35])

# visualize clusters


# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'

data=pd.merge(clusters,transactions)

# merge `data` and `offers`
data=pd.merge(offers,data)

# initialzie empty dictionary
champagne={}

# iterate over every cluster
for i in range(0,5):
    # observation falls in that cluster
    new_df=data[data.cluster==i]
    # sort cluster according to type of 'Varietal'
    counts=new_df['Varietal'].value_counts(ascending=False)
    # check if 'Champagne' is ordered mostly
    if counts.index[0]=='Champagne':
        # add it to 'champagne'
        champagne.update({i:counts[0]})

# get cluster with maximum orders of 'Champagne' 
# cluster_champagne==max(champagne)
cluster_champagne=max(champagne, key=champagne.get)
# print(counts)
# cluster_champagne=0
# print out cluster number
print(cluster_champagne)



# --------------
# Code starts here

# empty dictionary
discount={}

# iterate over cluster numbers
for val in data.cluster.unique():
    # dataframe for every cluster
    new_df = data[data.cluster == val]
    # average discount for cluster
    counts=new_df['Discount (%)'].values.sum()/len(new_df)
    # adding cluster number as key and average discount as value 
    discount[val] = counts
# cluster with maximum average discount

cluster_discount = max(discount, key=discount.get)
print(cluster_discount)

# Code ends here


