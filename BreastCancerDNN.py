#
# Pythology Session
# September 2017
# Brandon Boynton
# Breast Cancer Identifier
#

import pandas as pd
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score

#Configuration variables
TEST_SPLIT = 0.1
HOLE_SYMBOL = '?'
PREDICTION_COL = 'class'
HM_EPOCHS = 10
LEARNING_RATE = 0.001
HM_INPUTS = 9
HM_OUTPUTS = 2

#
#   ~~ 1. preprocessing
#

df = pd.read_csv('breast-cancer-wisconsin.csv') #Load CSV into pandas Dataframe

#   1a. Drop any columns that are not pertinant
df.drop("id", axis=1, inplace=True) #Drop the ID column

#   1b. Replace any holes in the data with an unnoticable value
cols = list(df.columns.values) #Get all of the columns
for col in cols: #for each column
    only_nums = [] #Initialize array for all the numbers in the array
    for row in df[col].tolist(): #For every row in the column
        if isinstance(row, (int, long, float, complex)):
            only_nums.append(row) #Add all the numbers to only_nums
    col_mean = np.mean(only_nums) #Get the average of all the rows
    new_col = df[col].tolist() #Create new column with the existing column
    for row in new_col: #Loop through new column
        if row == HOLE_SYMBOL: #Replace any holes with the average value
            new_col[row] = col_mean
    df[col] = new_col #Replace existing column with new column

# In other cases, you may need to drop rows, convert strings into numerical data, or clense data

#    1c. Set up dataframe for the neural network
hm_test_rows = int(len(df.index) * float(TEST_SPLIT)) #How many rows should we reserve for testing the AI
hm_train_rows = len(df.index) - hm_test_rows #How many rows should we train on

train = df.head(hm_train_rows) #Get the first # of rows for training
test = df.tail(hm_test_rows) #Get the last # of rows for testing

X = np.array(train.drop([PREDICTION_COL],1).astype(float)) #Format input data into variable X
X = np.array(X).reshape(hm_train_rows, HM_INPUTS) #turn multidimensional array into readable shape
y = np.array(df[PREDICTION_COL]) #get only self.prediction_col

#   1d. Convert values into one-hot encoded tensors
label_vals = [2,4] #Possible outputs

new_y = []
for label in y: #for each value in y
    empty_tensor = [0,0] #create array with 0 for each unique element in y
    modified_tensor = np.array(empty_tensor) # create new modified_tensor from empty_tensor
    label_index = label_vals.index(label) # get the index of that element from all unique elements
    modified_tensor[label_index] = 1 #set that index to 1
    new_y.append(modified_tensor)
y = new_y #replace y with the new formatted y

test_X = np.array(test.drop([PREDICTION_COL],1).astype(float))
test_X = np.array(test_X).reshape(hm_test_rows, HM_INPUTS)
test_y = y[hm_train_rows:]

y = y[:hm_train_rows]
y = np.array(y)

y_true = test_y #format test_y labels for the acuracy test
y_true = [np.argmax(x) for x in y_true]

#
#   ~~ 2. Create the neural network architecture
#

net = Sequential() #Create model

import tensorflow as tf #Reset graph just in case
tf.reset_default_graph()
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.InteractiveSession()

net.add(Dense(HM_INPUTS,input_dim=HM_INPUTS))
#net.add(Dense(HM_INPUTS,input_shape=(None,HM_INPUTS)))
net.add(Dense(14))
net.add(Dense(HM_OUTPUTS))
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #For custom optimizer
net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) #Uses default optimizer

#
#   ~~ 3. Train the nerual network
#

net.fit(X, y, epochs=HM_EPOCHS)

#   3a. Get the accuracy

y_scores = net.predict(test_X)
y_scores = [np.argmax(x) for x in y_scores]
accuracy = accuracy_score(y_true, y_scores)

print("Network Accuracy: {}".format(accuracy))

#Create infinite loop for test inferencing
while (True):
    inference_str = raw_input('> ')
    inference_tensor = inference_str.split(',')
    inference_tensor = np.array(map(int, inference_tensor))
    inference_tensor = inference_tensor.reshape((1,HM_INPUTS))
    results = net.predict(inference_tensor)
    print(results)
