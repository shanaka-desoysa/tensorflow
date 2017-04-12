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
LEARN_RATE = 1.0e-3
NUM_OF_EPOCHS = 1000


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def analyse_fire_theft_regression(scale='linear',
                                  learning_rate=1.0e-3,
                                  num_of_epochs=100,
                                  data_file='data/fire_theft.xls',
                                  log_file='logs/fire_theft'):
    # Step 1: read in data from the .xls file
    book = xlrd.open_workbook(data_file, encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = sheet.nrows - 1

    # Normalize features
    all_xs = feature_normalize(data.T[0])
    all_xs = np.transpose([all_xs])
    all_ys = np.transpose([data.T[1]])

    # Step 2: create placeholders for input X (number of fire) and label Y
    # (number of theft)
    X = tf.placeholder(tf.float32, [None, 1], name='X')
    Y = tf.placeholder(tf.float32, [None, 1], name='Y')

    # Step 3: create weight and bias, initialized to 0
    W1 = tf.Variable(tf.zeros([1, 1]), name="weights")
    W2 = tf.Variable(tf.zeros([1, 1]), name="weights")
    W3 = tf.Variable(tf.zeros([1, 1]), name="weights")
    B = tf.Variable(tf.zeros([1]), name="bias")

    # Step 4: build model to predict Y
    with tf.name_scope("Wx_b") as scope:
        # Linear by default
        Y_predicted = X * W1 + B
        #Y_predicted = X ** 2 * W2 + X * W1 + B

        if scale == 'quadratic':
            Y_predicted = X ** 2 * W2 + X * W1 + B
        if scale == 'cubic':
            Y_predicted = X ** 3 * W3 + X ** 2 * W2 + X * W1 + B

    # Add summary ops to collect data
    W_hist = tf.summary.histogram("weight", [W1, W2, W3])
    B_hist = tf.summary.histogram("biases", B)
    Y_hist = tf.summary.histogram("y", Y_predicted)

    # Step 5: use the square error as the loss function
    with tf.name_scope("loss") as scope:
        loss = tf.reduce_mean(tf.square(Y - Y_predicted), name='loss')
        los_sum = tf.summary.scalar("loss", loss)

    # Step 6: using gradient descent with learning rate of 0.01 to minimize
    # loss
    '''
    LEARNING_RATE = LEARN_RATE

    if scale == 'quadratic':
        LEARNING_RATE = LEARN_RATE ** 2
    if scale == 'cubic':
        LEARNING_RATE = LEARN_RATE ** 3
    '''
    with tf.name_scope("train") as scope:
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss)

    # Calculate accuracy of the model using RMSE
    RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, Y_predicted))))

    sess = tf.Session()

    # Merge all the summaries and write them out to logs
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_file, sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    all_feed = {X: all_xs, Y: all_ys}

    for i in range(num_of_epochs):
        # Record summary data, and the accuracy every 10 steps
        if i % 10 == 0:
            result = sess.run(merged, feed_dict=all_feed)
            writer.add_summary(result, i)
        else:
            sess.run(train_step, feed_dict=all_feed)

        if i % 10 == 0:
            print("After %d iteration:" % i)
            #print("W: %s" % sess.run(W))
            #print("b: %f" % sess.run(B))
            print("loss: %f" % sess.run(loss, feed_dict=all_feed))
            print("RMSE: %f" % sess.run(RMSE, feed_dict=all_feed))

    print("loss: %f" % sess.run(loss, feed_dict=all_feed))
    print("RMSE: %f" % sess.run(RMSE, feed_dict=all_feed))

    # Step 9: output the values of w and b
    all_feed = {X: all_xs}

    pred_Y = sess.run(Y_predicted, feed_dict=all_feed)

    writer.flush()
    writer.close()
    sess.close()

    X, Y = all_xs, all_ys
    plt.plot(X, Y, 'bo', label='Real data')
    plt.plot(X, pred_Y, 'ro', label='Predicted data')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(Y, pred_Y)
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


#analyse_fire_theft_regression(scale='linear', learning_rate=1.0e-3, num_of_epochs=1000)
# RMSE: 19.898848

#analyse_fire_theft_regression(scale='quadratic', learning_rate=1.0e-3, num_of_epochs=1000)
# loss: 380.387451
# RMSE: 19.503525

analyse_fire_theft_regression(
scale='cubic', learning_rate=1.0e-3, num_of_epochs=1000)
# loss: 291.596008
# RMSE: 17.076183


# analyse_fire_theft_regression('quadratic')
# RMSE: 22.627415 @10,000

# analyse_fire_theft_regression('cubic')
# RMSE: 28.508011 @10,000
# RMSE: 27.787916 @20,000
