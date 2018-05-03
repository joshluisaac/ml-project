import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


data_path="/home/joshua/Desktop/datascience/data_analysis_2/etl_load_history_group9k.csv"

#data = np.loadtxt("etl_transaction_yyc.csv", delimiter="|")

data = np.loadtxt(data_path, delimiter="|")



k=2
kmeans = KMeans(n_clusters=k)

kmeans.fit(data)

centriods = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centriods: {}".format(centriods))
print("Labels: {}".format(labels))

#pred = kmeans.predict([[83197, 56], [93000, 30]])


#print(pred)
#print ("Predicted labels: {}".format(pred))


# Graphing

xAxis=data[:,0]
yAxis=data[:,1]

plt.scatter(xAxis,yAxis, label='True Position')

plt.title('Number of records processed vs Throughput/sec')
plt.xlabel('Number of records processed/Payload')
plt.ylabel('Throughput/sec')

plt.scatter(data[:,0],data[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(centriods[:,0] ,centriods[:,1], color='black')

plt.show()
