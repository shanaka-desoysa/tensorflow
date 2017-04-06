# pylint: disable=invalid-name


'''
Problem 1. 
Please considered attached Excel file called Reduced_Car_Data.xlsx. 
This is the data set we used previously except that we have now removed several 
descriptive variables and left only: Displacement, Horsepower and Weight. 
Please build a regression model using TensorFlow that will predict the gasoline 
consumption(MPG - Miles Per Gallon) of cars based on three remaining variables. 
Please extract a percentage of data to serve as a training set and a percentage 
to serve as the test set. Please report on the accuracy of your model.
'''

# Import tensorflow and other libraries.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

#%matplotlib inline
import pylab
import pandas as pd
import xlrd

DATA_FILE = 'data/Reduced_Car_Data.xlsx'
LOG_FILE = 'logs/Reduced_Car_Data'
n_epochs = 100000
learn_rate = 0.00000001

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Print number of data points
print(n_samples)
np.random.seed(1234)
msk = np.random.rand(len(data)) < 0.75
train = data[msk]
test = data[~msk]
print('training/test data set length: {0}/{1}'.format(len(train), len(test)))

# Step 2: create placeholders for input X (number of fire) and label Y
# (number of theft)
X = tf.placeholder(tf.float32, [None, 3], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

# Step 3: create weight and bias, initialized to 0
W = tf.Variable(tf.zeros([3, 1]), name="weights")
B = tf.Variable(tf.zeros([1]), name="bias")

# Step 4: build model to predict Y
with tf.name_scope("Wx_b") as scope:
    product = tf.matmul(X, W)
    Y_predicted = product + B

# Add summary ops to collect data
W_hist = tf.summary.histogram("weight", W)
B_hist = tf.summary.histogram("biases", B)
Y_hist = tf.summary.histogram("y", Y_predicted)

# Step 5: use the square error as the loss function
# Cost function sum((y_-y)**2)
with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.square(Y - Y_predicted))
    cost_sum = tf.summary.scalar("cost", cost)


# Step 6: using gradient descent with learning rate of 0.01 to minimize
# loss
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001).minimize(loss)
with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

# Calculate accuracy of the model using RMSE
RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, Y_predicted))))

# Traing dataset
all_Xs = train[:, 1:4]
all_Ys = train[:, 4]
all_Xs = np.array(all_Xs)
all_Ys = np.transpose([all_Ys])

all_feed = {X: all_Xs, Y: all_Ys}

sess = tf.Session()

# Merge all the summaries and write them out to logs
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(n_epochs):
    # Record summary data, and the accuracy every 10 steps
    if i % 10 == 0:
        result = sess.run(merged, feed_dict=all_feed)
        writer.add_summary(result, i)
    else:
        sess.run(train_step, feed_dict=all_feed)

    if i % 1000 == 0:
        print("After %d iteration:" % i)
        #print("W: %s" % sess.run(W))
        #print("b: %f" % sess.run(B))
        print("cost: %f" % sess.run(cost, feed_dict=all_feed))
        print("RMSE: %f" % sess.run(RMSE, feed_dict=all_feed))

W_value, B_value = sess.run([W, B])

print(W_value, B_value)
# close the writer when you're done using it
writer.flush()
writer.close()

# Get predictions for Test dataset
all_Xs = test[:, 1:4]
all_Ys = test[:, 4]
all_Xs = np.array(all_Xs)
all_Ys = np.transpose([all_Ys])

all_feed = {X: all_Xs, Y: all_Ys}

Y_hat = sess.run(Y_predicted, feed_dict=all_feed)
RMSE_Test = sess.run(RMSE, feed_dict=all_feed)

# Close the session
sess.close()

square = np.square(all_Ys - Y_hat)
#print("Sum of square errors: {0}".format(np.sum(square)))
#print("Root sum of square errors: {0}".format(math.sqrt(np.sum(square))))
print("Accuracy of the model, RMSE: {0}".format(RMSE_Test))

plt.plot(range(1, len(test) + 1), all_Ys, 'bo', label='Real data')
plt.plot(range(1, len(test) + 1), Y_hat, 'r', label='Predicted data')
plt.legend()
plt.show()
