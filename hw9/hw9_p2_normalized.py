"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of fire in the city of Chicago
"""
# pylint: disable=invalid-name

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'
LOG_FILE = 'logs/fire_theft'
LEARNING_RATE = 0.0001
NUM_EPOCH = 500

# Standardize X values.
# Calculate Z score with mea = 0, sd = 1


def feature_standardize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Normalize X values
all_X, all_Y = data.T[0], data.T[1]
all_X = feature_standardize(all_X)

all_X = np.transpose([all_X])
all_Y = np.transpose([all_Y])

# Step 2: create placeholders for input X (number of fire) and label Y
# (number of theft)
X = tf.placeholder(tf.float32, [None, 1], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

# Step 3: create weight and bias, initialized to 0
w1 = tf.Variable(tf.zeros([1, 1]), name='weights1')
w2 = tf.Variable(tf.zeros([1, 1]), name='weights2')
w3 = tf.Variable(tf.zeros([1, 1]), name='weights3')

b = tf.Variable(tf.zeros([1]), name='bias')

# Step 4: build model to predict Y
#Y_predicted = X * w1 + b
#Y_predicted = X ** 2 * w2 + X * w1 + b
Y_predicted = X ** 3 * w3 + X ** 2 * w2 + X * w1 + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')
loss_sum = tf.summary.scalar("loss", loss)
mean_loss = tf.reduce_mean(tf.square(Y - Y_predicted), name='mean_loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

sess = tf.Session()
# Merge all the summaries and write them out to log file
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

# Step 7: initialize the necessary variables, in this case, w and b
sess.run(tf.global_variables_initializer())

all_feed={X: all_X, Y: all_Y}
# Step 8: train the model
for i in range(NUM_EPOCH):  # train the model 100 times
    total_loss = 0
    _, l = sess.run([optimizer, loss], feed_dict=all_feed)
    total_loss = l.sum()

    #print('Epoch {0}: Total lost: {1}'.format(i, total_loss))
    #print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    print('Epoch {0}: Mean: {1}'.format(i, sess.run(mean_loss, feed_dict={X: all_X, Y: all_Y})))

# close the writer when you're done using it
writer.close()

# Step 9: output the values of w and b
Y_pred = sess.run(Y_predicted, feed_dict={X: all_X, Y: all_Y})

sess.close()
# plot the results
X, Y = all_X, all_Y
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, Y_pred, 'ro', label='Predicted data')
plt.legend()
plt.show()

'''
LEARNING_RATE = 0.0001
NUM_EPOCH = 100
Linear Epoch 99: Mean: 360.8202209472656
Quad Epoch 99: Mean: 333.27801513671875
Cube Epoch 999: Mean: 143.69219970703125
Cube Epoch 499: Mean: 197.4280548095703
'''
