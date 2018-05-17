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

file_path_url = "/media/joshua/martian/staffordshireUniversity/phd-thesis/datafiles/customerInvoiceDataPred.vec"
out_path = "/media/joshua/martian/staffordshireUniversity/mlthesis/out"
predicted_data_name = "data_pred.csv"

def read_prediction_dataset(file_path):
    """Loads the prediction data set and returns a numpy array
    Assign column names to the dataset
    Read prediction data set
    """
    names = ['CustomerId_PKEY','CustomerName_OPT','InvoiceDate_NN','InvoiceStatus_OPT','label']
    dataset_prediction = pd.read_csv(file_path, sep='|', names=names, nrows=5000)
    
    dataset_prediction.describe().to_json(out_path + "/" + "gaussian_pred.json")
    return dataset_prediction.iloc[:, :-1].values


#load model for reuse using pickle
model_file_path = '/media/joshua/martian/staffordshireUniversity/phd-thesis/models'
model_name = 'gaussian_finalized_model.sav'
model_file_name = model_file_path + '/' + model_name
#print type(loaded_model)

def load_saved_model(filename):
    return pickle.load(open(filename, 'rb'))


def predict(model,data):
    """Will iterate over a numpy array and assign each row to a list
    That list is the prediction result which will be later serialized to csv file
    """
    prediction_list = []
    for row in data:
        rowFmt = "{}|{}|{}|{}".format(row[0],row[1],row[2],row[3])
        prediction_row_result = model.predict([row])
        fmtRow = "{}|{}\n".format(rowFmt,prediction_row_result[0])
        prediction_list.append(fmtRow)
    return prediction_list


def stream_prediction_data_to_csv(pred_data, csv_file_name):
    with open(csv_file_name, 'wb') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(pred_data)


def get_pred_metrics(model_type,pred_data):
    prediction_map = {}
    prediction_map['prediction_file_length'] = len(pred_data)
    prediction_map['model_type'] = "sklearn.naive_bayes.GaussianNB"
    prediction_map['model_name'] = model_name
    return prediction_map


def run_app():
    prediction_data = read_prediction_dataset(file_path_url)
    loaded_model = load_saved_model(model_file_name)
    pred_data = predict(loaded_model,prediction_data)
    stream_prediction_data_to_csv(pred_data,out_path +"/"+ predicted_data_name)
    result = get_pred_metrics(loaded_model,pred_data)
    return result



if __name__ == '__main__':
    run_app()



