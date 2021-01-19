import pandas as pd
import numpy as np
import geopandas as gpd
import math
import matplotlib.pyplot as plt

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


email_tail_list = []
email_series = data["Email"]

for i in email_series:
    head, sep, tail = i.partition("@")
    email_tail_list.append(tail)

email_tail_series = pd.Series(email_tail_list)

tail_cat_list = []

for i in email_tail_list:
    if i == "gmail.com":
        tail_cat_list.append(1)
    if i == "hotmail.com":
        tail_cat_list.append(2)
    if i == "yahoo.com":
        tail_cat_list.append(3)
    else:
        tail_cat_list.append(4)

email_new = pd.Series(tail_cat_list)

#print(email_new.head(15))
print(email_new.value_counts())