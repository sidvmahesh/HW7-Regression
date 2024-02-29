"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from regression.logreg import BaseRegressor, LogisticRegressor
from regression.utils import loadDataset
# (you will probably need to import more things here)

def test_prediction():
	X_train, X_val, y_train, y_val = loadDataset(split_percent = 0.8)
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	lr = LogisticRegressor(6, learning_rate = 0.1, max_iter = 1, tol = 0.0001)
	input = X_train[0]
	assert_almost_equal(np.clip(lr.make_prediction(input), 1e-10, 1 - 1e-10), lr.make_prediction(input))

def test_loss_function():
	X_train, X_val, y_train, y_val = loadDataset(split_percent = 0.8)
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	lr = LogisticRegressor(6, learning_rate = 0.1, max_iter = 1, tol = 0.0001)
	input = X_train[0]
	y_pred = np.clip(lr.make_prediction(input), 1e-10, 1 - 1e-10)
	y_true = y_train[0]
	assert_almost_equal(np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)), lr.loss_function(y_true = y_true, y_pred = y_pred))

def test_gradient():
	X_train, X_val, y_train, y_val = loadDataset(split_percent = 0.8)
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	lr = LogisticRegressor(6, learning_rate = 0.1, max_iter = 2, tol = 0.0001)
	input = X_train[0:1]
	y_pred = np.clip(lr.make_prediction(input), 1e-10, 1 - 1e-10)
	y_true = y_train[0:1]
	assert_almost_equal(np.matmul(y_pred - y_true, input), lr.calculate_gradient(y_true = y_true, X = input))

def test_training():
	X_train, X_val, y_train, y_val = loadDataset(split_percent = 0.8)
	lr = LogisticRegressor(6, learning_rate = 0.1, max_iter = 10000, tol = 0.0001)
	lr.train_model(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val)
	# Loss curve looks good, but not plotting so that it runs on GitHub Actions CI/CD
	#lr.plot_loss_history()