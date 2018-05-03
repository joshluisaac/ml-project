import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


data_path = "/home/joshua/Desktop/datascience/data_analysis_2/sample_knn.csv"

data = pd.read_csv(data_path, sep='|')

#data = np.loadtxt("etl_transaction_yyc.csv", delimiter="|")

#data = np.loadtxt(data_path, delimiter="|")

print data


X = data.iloc[:, :4].values


print X

k=5
kmeans = KMeans(n_clusters=k)

kmeans.fit(X)

centriods = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centriods: {}".format(centriods))
print("Labels: {}".format(labels))