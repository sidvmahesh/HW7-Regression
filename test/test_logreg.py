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
from regression.logreg import BaseRegressor, LogisticRegressor
from regression.utils import loadDataset
# (you will probably need to import more things here)

def test_prediction():
	X_train, X_val, y_train, y_val = loadDataset(split_percent = 0.8)
	lr = LogisticRegressor(6, learning_rate = 0.1, max_iter = 10000, tol = 0.0001)
	lr.train_model(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val)
	lr.plot_loss_history()

def test_loss_function():
	pass

def test_gradient():
	pass

def test_training():
	pass