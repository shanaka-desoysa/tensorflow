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

# print(data
print(n_samples)


# Step 2: create placeholders for input X (number of fire) and label Y
# (number of theft)
X = tf.placeholder(tf.float32, [1, 3], name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable(name='weights', shape=[
                    3, 1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict Y
Y_predicted = tf.matmul(X, w) + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')
loss_sum = tf.summary.scalar("loss", loss)


# Step 6: using gradient descent with learning rate of 0.01 to minimize
# loss
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learn_rate).minimize(loss)

Y_hat = []
all_Xs = data[:, 1:4]
all_Ys = data[:, 4]
all_Xs = np.array(all_Xs)
all_Ys = np.transpose([all_Ys])

all_feed = {X: all_Xs, Y: all_Ys}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_FILE, sess.graph)
    for i in range(n_epochs):
        total_loss = 0
        for row, displacement, horsepower, weight, mpg in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: np.reshape(
                [displacement, horsepower, weight], (1, 3)), Y: mpg})
            total_loss += l
        if i % 10 == 0:
            print('Average loss epoch {0}: {1}'.format(i, total_loss))

    w_value, b_value = sess.run([w, b])

    print(w_value, b_value)

    # Predicted values
    for row, displacement, horsepower, weight, mpg in data:
        Y_hat.append(sess.run(Y_predicted, feed_dict={X: np.reshape(
            [displacement, horsepower, weight], (1, 3)), Y: mpg}))

Y_hat = np.array([Y_hat])
Y_hat = Y_hat.reshape(100, 1)
square = np.square(all_Ys - Y_hat)
print("sum of square errors: {0}".format(np.sum(square)))
print("root sum of square errors: {0}".format(math.sqrt(np.sum(square))))

#square = np.sqrt(square)
plt.plot(range(1, 101), all_Ys, 'bo', label='Real data')
plt.plot(range(1, 101), Y_hat, 'r', label='Predicted data')
#plt.plot(range(1, 101), square, 'go', label='Square')
plt.legend()
plt.show()


'''
[[-0.1131861 ]
 [-0.02056148]
 [ 0.01876079]] 0.00162297
sum of square errors: 19790.572905882957
root sum of square errors: 140.6789710862393
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
