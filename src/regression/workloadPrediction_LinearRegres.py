import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import module to calculate model perfomance metrics
from sklearn import metrics

import json

sns.set()


dataPath = "/media/joshua/martian/staffordshireUniversity/phd-thesis/datafiles/workload_prediction_data.csv"
colNames = ['Payload','RunningTime','ThroughputPersec']
#dataset = pd.read_csv(dataPath, delimiter="|", names=colNames, header=None)

def loadPredictionDataSet(dataPath, colNames):
    """Will load all the data points in the training data set returning a panda data frame
    This will be used for the model prediction
    """
    return pd.read_csv(dataPath, delimiter="|", names=colNames, header=None)


dataset = loadPredictionDataSet(dataPath,colNames)

#print dataset.head()
#print dataset.shape
statsSummary = dataset.describe()
#print type(statsSummary)

checkType = isinstance(statsSummary, pd.core.frame.DataFrame)

#print checkType

# write stats summary to JSON file
statsSummary.to_json("workloadmetrics.json")

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
print("Constant/Intercept: ", linreg.intercept_)

# coefficient of x, also known as the slope of the graph.
# y = mx + c
# Denoted as B1
print("Slope of the graph: ",linreg.coef_)

#print linreg.coef_[0]

df = pd.DataFrame({'Through put per sec': xTest1D, 'Actual': y_test, 'Predicted': y_pred}) 


#Evaluating the algorithm
# compute the RMSE of our predictions
rootMeanSquaredError = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
meanAbsoluteError = metrics.mean_absolute_error(y_test, y_pred)
meanSquaredError = metrics.mean_squared_error(y_test, y_pred)
print("MAE: {}".format(meanAbsoluteError))
print("MSE: {}".format(meanSquaredError))
print("RMSE: {}".format(rootMeanSquaredError))

def getMetrics():
    return [linreg.intercept_,linreg.coef_[0],meanAbsoluteError,meanSquaredError,rootMeanSquaredError]

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
