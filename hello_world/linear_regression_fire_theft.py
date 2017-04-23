
# pylint: disable=invalid-name

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'
LOG_FILE = 'logs/fire_theft'
learn_rate = 0.001
num_epoch = 100

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
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict Y
Y_predicted = X * w + b

# Add summary ops to collect data
W_hist = tf.summary.histogram("weights", w)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y", Y_predicted)

# Step 5: use the square error as the loss function
#loss = tf.square(Y - Y_predicted, name='loss')
#loss_sum = tf.summary.scalar("loss", loss)
cost = tf.reduce_mean(tf.square(Y - Y_predicted))
cost_sum = tf.summary.scalar("cost", cost)

# Step 6: using gradient descent with learning rate of 0.01 to minimize
# loss
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learn_rate).minimize(cost)
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    # Merge all the summaries and write them out to logs
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

    # Step 8: train the model
    for i in range(num_epoch):  # train the model 100 times
        total_cost = 0
        for x, y in data:
            # Session runs train_op and fetch values of loss
            _, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y})
            total_cost += c
            print('Epoch {0}: {1}'.format(i, total_cost / n_samples))

            if i % 10 == 0:
                all_feed = {X: data.T[0], Y: data.T[1]}
                result = sess.run(merged, feed_dict=all_feed)
                writer.add_summary(result, i)

    # close the writer when you're done using it
    writer.flush()
    writer.close()

    # Step 9: output the values of w and b
    w_value, b_value = sess.run([w, b])

sess.close()

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
