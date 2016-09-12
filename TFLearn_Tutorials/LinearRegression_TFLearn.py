""" Linear Regression Example """

from __future__ import absolute_import, division, print_function

import tflearn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_train_data():
	with open("/mnt/d/TF_ML/EmploymentWages.txt", "r") as fp:
		emp_data = fp.readlines()
	
	X = []
	Y = []
	cnt = 1
	for line in emp_data:
		line = line.strip()
		arr = line.split('\t')
		Y.append(int(arr[1]))
		X.append(cnt)
		cnt += 1
		#X.append(int(arr[0]))
	
	return X, Y
	
X, Y = get_train_data()

X1, X2 = 15, 16

# Linear Regression graph
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.05)
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=25000, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")
print("Y = " + str(m.get_weights(linear.W)) +
      ".X + " + str(m.get_weights(linear.b)))

print("\nTest prediction for x1 = %d and x2 = %d:" % (X1, X2))
Y1, Y2 = m.predict([X1, X2])
print(Y1, Y2)