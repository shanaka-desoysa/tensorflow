"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'
LOG_FILE = 'logs/fire_theft'
LEARNING_RATE = 0.000000001
NUM_EPOCH = 6000

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y
# (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf . Variable(0.0, name="weights_1")
u = tf . Variable(0.0, name="weights_2")
c = tf . Variable(0.0, name="weights_3")
b = tf . Variable(0.0, name="bias")

# Step 4: build model to predict Y
Y_predicted = (X * X * X * c) + (X * X * w) + (X * u) + b

# Step 5: use the square error as the loss function
#loss = tf.square(Y - Y_predicted, name='loss')
loss = tf.sqrt(tf.reduce_mean(tf.square(Y - Y_predicted)), name='loss')
# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

    # Step 8: train the model
    for i in range(NUM_EPOCH):  # train the model 100 times
        total_loss = 0
        for x, y in data:
            # Session runs train_op and fetch values of loss
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w_value, u_value, c_value, b_value = sess.run([w, u, c, b])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, (X * X * X * c_value) + (X * X * w_value) + (X * u_value) + b_value, 'ro', label='Predicted data')
plt.legend()
plt.show()

'''
LEARNING_RATE = 0.0000001
NUM_EPOCH = 3000
Epoch 2999: 20.13805685724531
Epoch 4999: 22.906322536014375

LEARNING_RATE = 0.000000001
NUM_EPOCH = 3000
Epoch 2999: 25.8672419048491
Epoch 5999: 25.643278712318057
'''
