# pylint: disable=invalid-name

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
n_epochs = 500
learn_rate = 0.00000001

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

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

all_Xs = data[:, 1:4]
all_Ys = data[:, 4]
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

    print("After %d iteration:" % i)
    #print("W: %s" % sess.run(W))
    #print("b: %f" % sess.run(B))
    print("cost: %f" % sess.run(cost, feed_dict=all_feed))

W_value, B_value = sess.run([W, B])

print(W_value, B_value)
# close the writer when you're done using it
writer.flush()
writer.close()

Y_hat = sess.run(Y_predicted, feed_dict=all_feed)
square = np.square(all_Ys - Y_hat)
print("sum of square errors: {0}".format(np.sum(square)))
print("root sum of square errors: {0}".format(math.sqrt(np.sum(square))))

plt.plot(range(1, 101), all_Ys, 'bo', label='Real data')
plt.plot(range(1, 101), Y_hat, 'r', label='Predicted data')
plt.legend()
plt.show()


sess.close()

'''
[[-0.00439648]
 [-0.00092727]
 [ 0.00701894]] [  3.10274445e-05]
sum of square errors: 16923.49630659737
root sum of square errors: 130.09033902099483
'''

'''with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_FILE, sess.graph)
    for i in range(n_epochs):


        total_loss = 0
        for row, displacement, horsepower, weight, mpg in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: np.reshape(
                [displacement, horsepower, weight], (1, 3)), Y: mpg})
            total_loss += l
        print('Average loss epoch {0}: {1}'.format(i, total_loss))

    W_value, B_value = sess.run([W, B])

    print(W_value, B_value)
'''

'''with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    # Merge all the summaries and write them out to logs
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

    # Step 8: train the model
    for i in range(100):  # train the model 100 times
        total_cost = 0
        for x1, x2, x3, x4, y in data:
            # Session runs train_op and fetch values of loss
            _, c = sess.run([optimizer, cost], feed_dict={X1: x2, X2: x3, Y: y})
            total_cost += c
            #print('Epoch {0}: {1}'.format(i, total_cost / n_samples))
            #print("y: %s" % sess.run(y, feed_dict=feed))
            #print("y_: %s" % ys)
            #print("cost: %f" % sess.run(cost, feed_dict=feed))
            #print("After %d iteration:" % i)
            print("w1: %f" % sess.run(w1))
            print("w2: %f" % sess.run(w2))
            print("b: %f" % sess.run(b))

            if i % 10 == 0:
                all_feed = {X1: data.T[2], X2: data.T[3], Y: data.T[4]}
                result = sess.run(merged, feed_dict=all_feed)
                writer.add_summary(result, i)

    # close the writer when you're done using it
    writer.flush()
    writer.close()

    # Step 9: output the values of w and b
    #w1_value, w2_value, b_value = sess.run([w1, w2, b])
    #print(w1_value)
sess.close()
'''
