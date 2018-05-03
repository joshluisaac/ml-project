import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB

url = "/home/joshua/Desktop/datascience/data_analysis_2/sample_knn.csv"

# Assign colum names to the dataset
names = ['CustomerId','CustomerName','InvoiceDate','InvoiceStatus','label']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, sep='|', names=names)


clf = GaussianNB()



X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

clf.fit(X,y)
pred = clf.predict([[1,2,1,0]])

print pred


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


print X_test
print y_pred