# load data
import numpy as np
dataset = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1)

# shuffle dataset
np.random.shuffle(dataset)

# split data into X and y
X = dataset[:, 0:8]
y = dataset[:, 8]

# split test and training data (test : train = 10% : 90%)
test_data_number = int(dataset.shape[0] * 0.1)
train_data_number = dataset.shape[0] - test_data_number
test_X, train_X = X[:test_data_number], X[test_data_number:]
test_y, train_y = y[:test_data_number], y[test_data_number:]

# fit model on training data
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(train_X, train_y)
train_y_hat = model.predict(train_X)
print('>> train error:',
      np.linalg.norm(train_y_hat - train_y, ord=1) / train_data_number)

# test model
test_y_hat = model.predict(test_X)
print('>> test error:',
      np.linalg.norm(test_y_hat - test_y, ord=1) / test_data_number)

# plot feature importance
from xgboost import plot_importance
plot_importance(model)
from matplotlib import pyplot
pyplot.show()
