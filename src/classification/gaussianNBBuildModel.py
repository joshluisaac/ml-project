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
predUrl = "../../datafiles/customerInvoiceDataPred.vec"

# Assign colum names to the dataset
names = ['CustomerId_PKEY','CustomerName_OPT','InvoiceDate_NN','InvoiceStatus_OPT','label']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, sep='|', names=names)

#Read prediction data set
dataset_pred = pd.read_csv(predUrl, sep='|', names=names)

Model = GaussianNB()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

x_pred = dataset_pred.iloc[:, :-1].values

features_train, features_test, target_train, target_test = train_test_split(X, y, test_size = 0.33, random_state = 10)

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
#print dataset.describe()
print accuracy

#Model persistence using pickle
filename = 'gaussian_finalized_model.sav'
pickle.dump(Model, open(filename, 'wb'))

#load model for reuse
loaded_model = pickle.load(open(filename, 'rb'))
#print type(loaded_model)

#<type 'numpy.ndarray'>
#print type(x_pred) 

# iterate over the numpy array
pred_list = []
for row in x_pred:
    rowFmt = "{}|{}|{}|{}".format(row[0],row[1],row[2],row[3])
    predResult = loaded_model.predict([row])
    fmtRow = "{}|{}\n".format(rowFmt,predResult[0])
    pred_list.append(fmtRow)
    

def get_pred_metrics():
    prediction_map = {}
    file_length = len(pred_list)
    prediction_map['prediction_file_length'] = file_length
    print json.dumps(list(file_length))


def stream_prediction_data_to_file():
    with open('dataPred.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(pred_list)



#print loaded_model.predict(x_pred)
#result = loaded_model.score(features_test, target_test)
#print(result)

