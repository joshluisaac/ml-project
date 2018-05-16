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

file_path_url = "../../datafiles/customerInvoiceDataPred.vec"

def read_prediction_dataset(file_path):
    """Loads the prediction data set and returns a numpy array
    Assign column names to the dataset
    Read prediction data set
    """
    names = ['CustomerId_PKEY','CustomerName_OPT','InvoiceDate_NN','InvoiceStatus_OPT','label']
    dataset_prediction = pd.read_csv(file_path, sep='|', names=names)
    return dataset_prediction.iloc[:, :-1].values


#load model for reuse using pickle
model_file_name = 'gaussian_finalized_model.sav'
#print type(loaded_model)

def load_saved_model(filename):
    return pickle.load(open(filename, 'rb'))



# iterate over the numpy array and persist to list
prediction_list = []
def predict(model,data):
    for row in data:
        rowFmt = "{}|{}|{}|{}".format(row[0],row[1],row[2],row[3])
        prediction_row_result = model.predict([row])
        fmtRow = "{}|{}\n".format(rowFmt,prediction_row_result[0])
        prediction_list.append(fmtRow)


def get_pred_metrics():
    prediction_map = {}
    file_length = len(prediction_list)
    prediction_map['prediction_file_length'] = file_length
    print json.dumps(list(file_length))


def stream_prediction_data_to_file():
    with open('dataPred2.csv', 'wb') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(prediction_list)


def run_app():
    prediction_data = read_prediction_dataset(file_path_url)
    loaded_model = load_saved_model(model_file_name)
    predict(loaded_model,prediction_data)
    stream_prediction_data_to_file()
    return 0



if __name__ == '__main__':
    run_app()



