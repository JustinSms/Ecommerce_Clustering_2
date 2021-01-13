import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv("Ecommerce Customers.csv")
#print(data.head(5))

data_clustering = data[["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent"]]
print(data_clustering.head(5))


scaled = StandardScaler(data_clustering)
#print(data_clustering.head(5))


clustering = KMeans(n_clusters=5, random_state=20, max_iter=300)
data_clustering["clusters"] = clustering.fit_predict(data_clustering)

#print(data_clustering.head(10))


reduced_data = PCA(n_components= 2).fit_transform(data_clustering)
results = pd.DataFrame(reduced_data, columns= ["pca1","pca2"])

sns.scatterplot(x="pca1",y="pca2",hue=data_clustering["clusters"], data=results)
plt.show()


counter = 0

cluster_0 = []
cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []


for x in data_clustering["clusters"]:
    if x == 0:
        cluster_0.append(data_clustering.iloc[counter,:])
        counter += 1
    elif x == 1:
        cluster_1.append(data_clustering.iloc[counter,:])
        counter += 1
    elif x == 2:
        cluster_2.append(data_clustering.iloc[counter,:])
        counter += 1
    elif x == 3:
        cluster_3.append(data_clustering.iloc[counter,:])
        counter += 1
    elif x == 4:
        cluster_4.append(data_clustering.iloc[counter,:])
        counter += 1
    else:
        counter += 1
        pass

cluster_0 = pd.DataFrame(cluster_0)
cluster_1 = pd.DataFrame(cluster_1)
cluster_2 = pd.DataFrame(cluster_2)
cluster_3 = pd.DataFrame(cluster_3)
cluster_4 = pd.DataFrame(cluster_4)


#print(cluster_3.head(5))


print(cluster_3.describe())
print(cluster_0.describe())