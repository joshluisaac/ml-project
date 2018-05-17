import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import csv


# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle


url = "../../datafiles/customer_payment_data.csv"

dataPath = "/media/joshua/martian/staffordshireUniversity/phd-thesis/datafiles"
data_file_name = "customer_payment_data.csv"
model_file_path = '/media/joshua/martian/staffordshireUniversity/phd-thesis/models'
model_name = 'gaussian_finalized_model2.sav'

out_path = "/media/joshua/martian/staffordshireUniversity/mlthesis/out"
train_json = "gaussian_train.json"

# Assign colum names to the dataset
names = ['CustomerId_PKEY','CustomerName_OPT','InvoiceDate_NN','InvoiceStatus_OPT','label']

# Read dataset to pandas dataframe
dataset = pd.read_csv(dataPath +"/"+ data_file_name, sep='|', names=names)

Model = GaussianNB()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

test_size=0.33
random_state=10
features_train, features_test, target_train, target_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

#gnbClf.fit(X,y)
#pred = gnbClf.predict([[1,1,1,1]])

model = Model.fit(features_train,target_train)
target_pred = Model.predict(features_test)

print("New data point",Model.predict([[2,0,0,1]]))
print("New data point",Model.predict([[1,1,1,0]]))
print("New data point",Model.predict([[1,1,1,2]]))

accuracy = accuracy_score(target_test, target_pred, normalize = True)

#print features_test
#print target_pred

json_path = out_path + "/" + train_json
dataset.describe().to_json(json_path)
print "Created " + json_path
print accuracy

#Model persistence using pickle
pickle_path = model_file_path +"/"+ model_name
pickle.dump(Model, open(pickle_path, 'wb'))
print "Created " + pickle_path


gnb_train_stat = {}
gnb_train_stat["accuracy"] = accuracy
gnb_train_stat["test_size"] = test_size
gnb_train_stat["random_state"] = random_state


with open(out_path +"/"+ 'gnb_train_stats.json', 'w') as outfile:
    json.dump(gnb_train_stat, outfile)


