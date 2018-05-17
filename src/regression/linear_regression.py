import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import module to calculate model perfomance metrics
from sklearn import metrics
import pickle

import json

sns.set()


dataPath = "/media/joshua/martian/staffordshireUniversity/phd-thesis/datafiles/workload_prediction_data.csv"
model_file_path = '/media/joshua/martian/staffordshireUniversity/phd-thesis/models'
model_name = 'regres_finalized_model.sav'
out_path = "/media/joshua/martian/staffordshireUniversity/mlthesis/out" 
workload_pred = "workload_pred.json"
colNames = ['Payload','RunningTime','ThroughputPersec']
#dataset = pd.read_csv(dataPath, delimiter="|", names=colNames, header=None)

def load_dataset(dataPath, colNames):
    """Will load all the data points in the training data set returning a panda data frame
    This will be used for the model prediction
    """
    return pd.read_csv(dataPath, delimiter="|", names=colNames, header=None)


dataset = load_dataset(dataPath,colNames)


def get_dataframe_head(n):
    """ Returns first n rows of the DataFrame"""
    return dataset.head(n)

def get_dataframe_tail(n):
    """ Returns last n rows of the DataFrame"""
    return dataset.tail(n)

def get_dataframe_shape():
    """Returns the number of rows and columns"""
    return dataset.shape()

def get_summary():
    """Returns the statistical metrics"""
    return dataset.describe()


stats_summary = get_summary()


checkType = isinstance(stats_summary, pd.core.frame.DataFrame)


# Stream stats summary to JSON formatted file
json_out = out_path + "/" + workload_pred
stats_summary.to_json(json_out)
print "Created file " + json_out

#select all the rows of the first and third columns/attributes
# through put per sec
# x-axis is expected to be a 2D array and not 1D array
xAxis = dataset.iloc[:,[2]].values

# running time
yAxis =  dataset.iloc[:,1].values

#print dataset.Payload
#print dataset["Payload"]

# confirmation that what is loaded is a panda data frame type
#print type(dataset)


# Splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(xAxis, yAxis, test_size=0.2, random_state=0)

#print X_train
#print y_train

# Linear Regression Model
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)


# make predictions on the testing set
y_pred = linreg.predict(X_test)
#print X_test
#print y_pred

# returns a 1D view of X_test
xTest1D = X_test.ravel()


#intercept also known as bias B0
intercept = linreg.intercept_
print("Constant/Intercept: ", intercept)

# coefficient of x, also known as the slope of the graph.
# y = mx + c
# Denoted as B1
xCoef = linreg.coef_
print("Slope of the graph: ",xCoef)


df = pd.DataFrame({'Through put per sec': xTest1D, 'Actual': y_test, 'Predicted': y_pred}) 


#Evaluating the algorithm
# compute the RMSE of our predictions
rootMeanSquaredError = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
meanAbsoluteError = metrics.mean_absolute_error(y_test, y_pred)
meanSquaredError = metrics.mean_squared_error(y_test, y_pred)
print("MAE: {}".format(meanAbsoluteError))
print("MSE: {}".format(meanSquaredError))
print("RMSE: {}".format(rootMeanSquaredError))

def get_metrics():
    metrics_dict = {}
    metrics_dict['intercept'] = intercept
    metrics_dict['coefficient'] = xCoef[0]
    metrics_dict['mae'] = meanAbsoluteError
    metrics_dict['mse'] = meanSquaredError
    metrics_dict['rms'] = rootMeanSquaredError
    #[intercept,xCoef[0],meanAbsoluteError,meanSquaredError,rootMeanSquaredError]
    return metrics_dict


#dataset.plot(x="Throughput/sec", y="RunningTime", style=".")

#plt.scatter(xAxis,yAxis, label='True Position', alpha=1)

# use the function regplot to make a scatterplot
#sns.regplot(x=dataset["Throughput/sec"], y=dataset["RunningTime"])
#sns.regplot(x="Throughput/sec", y="RunningTime", data=dataset)
# same as
#sns.regplot(x=xAxis, y=yAxis, marker="*")
sns.regplot(x=xTest1D, y=y_pred, marker="o")
sns.regplot(x=xTest1D, y=y_test, color="green", marker="*")

#sns.lmplot(x=xTest1D, y=y_pred, hue=y_test, markers=["o", "x"])
#sns.pairplot(df, x_vars=[df.Actual], y_vars=[df.Predicted])

#plt.title('Payload vs Running Time')
plt.xlabel('Payload through put per sec')
plt.ylabel('Running Time (sec)')
#plt.show()

model_path = model_file_path + "/" + model_name
pickle.dump(linreg, open(model_path, 'wb'))
