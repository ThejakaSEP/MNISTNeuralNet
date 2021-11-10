import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/Users/thejakamahaulpatha/PycharmProjects/MNISTNeuralNet/train.csv')

# print(data.head(5))

data = np.array(data)
m,n = data.shape

# print(m,n)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

# print(X_train[:,0].shape)

