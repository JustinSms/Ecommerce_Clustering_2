import pandas as pd
import numpy as np

data = pd.read_csv("Ecommerce Customers.csv")

data_num = data[["Avg. Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent"]]

instance_norm_list = []


# Normalization session length
class Normalization:

    def __init__(self, min_v, max_v, series):
        self.min_v = min_v
        self.max_v = max_v
        self.series = series
        self.list = []

    def normalizator(self):
        for value in self.series:
            norm_value = ((value - self.min_v)/(self.max_v - self.min_v))

            self.list.append(norm_value)
        
        series_norm = pd.Series(self.list)

        return series_norm 


for i in data_num:
    instance = Normalization(data_num[i].min(), data_num[i].max(), data_num[i])
    instance_norm = instance.normalizator()
    instance_norm_list.append(instance_norm)


# new normalized dataframe
normalized_dataframe = pd.concat(instance_norm_list, axis=1)
normalized_dataframe.columns = list(data_num.columns)