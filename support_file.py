import pandas as pd
import numpy as np
import geopandas as gpd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes 
from sklearn.cluster import DBSCAN

data = pd.read_csv("Ecommerce Customers.csv")

data_num = data[["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent"]]

list_shortcut_states = ["AK","AL","AR","AZ","CA","CO","CT","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"]
state_list = []

for i in data["Address"]:
    if "Box" in i:
        index_to_drop = data[data["Address"] == i].index.values 
        data.drop(index_to_drop, inplace = True)
    else:
        state = i.split(",")[-1].split()[0]

        if state in list_shortcut_states:
            state_list.append(state)
        else:
            index_to_drop = data[data["Address"] == i].index.values 
            data.drop(index_to_drop, inplace = True)


data.drop(["Email","Avatar","Address"], inplace = True, axis = 1)
data.insert(5, "State", state_list)

data_mixed = data
data_mixed.index = range(len(data_mixed.index))

#print(data_mixed.head(5))
#print(data_mixed.State.value_counts())

state_group_list = []

for i in data_mixed["State"]:
    # 12 - 10
    if i in (["MO","DE","SC","OR","VT","FL","MS","MN","KS","NJ","NC"]):
        state_group_list.append("HIGH")
    # 9 - 8
    if i in (["AZ","HI","AL","MI","WV","ME","ND","NY","IL","TX","PA","GA","KY","MT"]):
        state_group_list.append("MEDIUM")
    # 7 - 0
    if i in (["MA","OK","WY","IN","IA","SD","AK","NH","RI","CA","NV","NE","VA","LA","NM","AR","WI","OH","CT","MD","CO","TN","UT","WA","ID"]):
        state_group_list.append("LOW")

state_group_series_new = pd.Series(state_group_list)

data_mixed_new = pd.concat([data_mixed, state_group_series_new], axis=1)



data_mixed_new.columns = ["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent","State","State group"]

#print(data_mixed_new.head(5))


hot_states = pd.get_dummies(data_mixed_new["State group"])
#print(hot_states.head(5))

data_hot_clustering = pd.concat([data_mixed_new,hot_states], axis=1)
#print(data_hot_clustering.head(5))



data_hot_clustering_only_dummies = data_hot_clustering[["HIGH","MEDIUM","LOW"]]

#print(data_hot_clustering_only_dummies.head(5))


#Import for kproto validation
data_stand = data_hot_clustering[["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent","State group"]]

con_feats = ["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent"]

scale = StandardScaler()

con_feats_scaled = scale.fit_transform(data_stand[con_feats])

con_feats_scaled_df = pd.DataFrame(con_feats_scaled)

data_stand = pd.concat([con_feats_scaled_df, data_hot_clustering["State group"]], axis=1)

#print(data_stand.head())

data_array=data_stand.values


#importing kproto for validation 
kproto = KPrototypes(n_clusters=3, max_iter=20)
clusters_proto = kproto.fit_predict(data_array, categorical=[5])

#print(kproto.cluster_centroids_)
#print(clusters_proto)

# DBSCAN
data_stand_DBSCAN = pd.concat([con_feats_scaled_df, data_hot_clustering[["HIGH","MEDIUM","LOW"]]], axis=1)

dbscan = DBSCAN(eps=1.6, min_samples=10)
clusters_dbscan = dbscan.fit_predict(data_stand_DBSCAN)
print(clusters_dbscan)