import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from kmodes import kprototypes
import seaborn as sns
import support_file as sf
from sklearn.preprocessing import StandardScaler


## Importing the data
data_hot_clustering = sf.data_hot_clustering
#data_hot_clustering = pd.concat([data_norm, dummies], axis=1)

data_hot_clustering = data_hot_clustering[["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent","State group", "HIGH","MEDIUM","LOW"]]
#print(data_hot_clustering.head(5))

# Standardized data
data_stand = data_hot_clustering[["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent","State group","HIGH","MEDIUM","LOW"]]

con_feats = ["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent"]
scale = StandardScaler()

data_stand[con_feats] = scale.fit_transform(data_stand[con_feats])
#print(data_stand.head())

##DBSCAN
X = data_stand.iloc[:,[1,4]].values

dbscan = DBSCAN(eps=0.3, min_samples=8)
clusters = dbscan.fit_predict(X)
labels_dbscan = dbscan.labels_
#print(labels_dbscan)

labels_df = pd.DataFrame(labels_dbscan)
labels_df.columns = ["Values"]

noise_list = []

index = labels_df.index
noise = labels_df["Values"] == -1
noise_indices = index[noise]

noise_list = noise_indices.tolist()

print(noise_list)


#np.delete(labels_dbscan, [i], 0)

"""
sns.scatterplot(X[:,0], X[:,1], hue=["cluster: {}".format(i) for i in labels_dbscan])
plt.xlabel("Avg. Session Lenght")
plt.ylabel("Yearly Amount Spent")
plt.title('DBSCAN clustering 2D' )
plt.legend(fancybox=False, fontsize='small')
plt.show()
"""












