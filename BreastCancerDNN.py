# Breast cancer identifier

import pandas as pd
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score

#Config variables
TEST_SPLIT = 0.1
HOLE_SYMBOL = '?'
PREDICTION_COL = 'class'
HM_EPOCHS = 10
LEARNING_RATE = 0.001

#
# ~~ 1. Preprocessing
#

df = pd.read_csv('breast-cancer-wisconsin.csv')
df.drop('id',axis=1,inplace=True)

cols = list(df.columns.values)
for col in cols:
	only_nums = []
	for row in df[col].tolist():
		if isinstance(row, (int, long, float, complex)):
			only_nums.append(row)
			col_mean = np.mean(only_nums)
			new_col = df[col].tolist()
			for row_str in new_col:
				if row_str == HOLE_SYMBOL:
					new_col[row] = col_mean
	df[col] = new_col

hm_test_rows = int(len(df.index) * float(TEST_SPLIT))
hm_train_rows = len(df.index) - hm_test_rows
hm_inputs = int(len(df.columns)-1)

train = df.head(hm_train_rows)
test = df.tail(hm_test_rows)

X = np.array(train.drop([PREDICTION_COL],1).astype(float))
X = np.array(X).reshape(hm_train_rows, hm_inputs)
y = np.array(df[PREDICTION_COL])

label_vals = [2,4]

new_y = []
for label in y:
	empty_tensor = [0,0]
	modified_tensor = np.array(empty_tensor)
	label_index = label_vals.index(label)
	modified_tensor[label_index] = 1
	new_y.append(modified_tensor)
y = new_y

test_X = np.array(test.drop([PREDICTION_COL],1).astype(float))
test_X = np.array(test_X).reshape(hm_test_rows, hm_inputs)
test_y = y[hm_train_rows:]

y = y[:hm_train_rows]
y = np.array(y)
hm_outputs = len(y[0])

y_true = test_y
y_true = [np.argmax(x) for x in y_true]

#
#  ~~ 2. Create the neural network
#

net = Sequential()

import tensorflow as tf
tf.reset_default_graph()
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.InteractiveSession()

net.add(Dense(hm_inputs, input_dim=hm_inputs))
net.add(Dense(14))
net.add(Dense(hm_outputs))

net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#
# ~~ Train
#

net.fit(X,y, epochs=HM_EPOCHS)

y_scores = net.predict(test_X)
y_scores = [np.argmax(x) for x in y_scores]
accuracy = accuracy_score(y_true, y_scores)

# inference

print("Network accuracy: {}".format(accuracy))

while(True):
	inference_str = raw_input('> ')
	inference_tensor = inference_str.split(',')
	inference_tensor = np.array(map(int, inference_tensor))
	inference_tensor = inference_tensor.reshape((1,hm_inputs))
	results = net.predict(inference_tensor)
	print(results)
