"""
Simple logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

"""
# pylint: disable=invalid-name

import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd

# Define paramaters for the model
learning_rate = 0.05  # 0.001, 0.005, 0.01, 0.02, 0.05
batch_size = 128  # 8, 64, 128, 256
n_epochs = 30
log_file = "logs/logistic_mnist"


def analyse_logistic_regression_mnist(learning_rate=0.01,
                                      batch_size=128,
                                      n_epochs=30,
                                      log_file="logs/logistic_mnist",
                                      debug=False):
    # Step 1: Read in data
    # using TF Learn's built in function to load MNIST data to the folder mnist
    mnist = input_data.read_data_sets('./mnist', one_hot=True)

    # Step 2: create placeholders for features and labels
    # each image in the MNIST data is of shape 28*28 = 784
    # therefore, each image is represented with a 1x784 tensor
    # there are 10 classes for each image, corresponding to digits 0 - 9.
    # each lable is one hot vector.
    X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder')
    Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y_placeholder')

    # Step 3: create weights and bias
    # w is initialized to random variables with mean of 0, stddev of 0.01
    # b is initialized to 0
    # shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
    # shape of b depends on Y
    w = tf.Variable(tf.random_normal(
        shape=[784, 10], stddev=0.01), name='weights')
    b = tf.Variable(tf.zeros([1, 10]), name="bias")

    # Step 4: build model
    # the model that returns the logits.
    # this logits will be later passed through softmax layer
    logits = tf.matmul(X, w) + b

    # Step 5: define loss function
    # use cross entropy of softmax of logits as the loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y, name='loss')
    # computes the mean over all the examples in the batch
    loss = tf.reduce_mean(entropy)

    # Step 6: define training op
    # using gradient descent with learning rate of 0.01 to minimize loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        # to visualize using TensorBoard
        writer = tf.summary.FileWriter(log_file, sess.graph)

        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        n_batches = int(mnist.train.num_examples / batch_size)
        for i in range(n_epochs):  # train the model n_epochs times
            total_loss = 0

            for _ in range(n_batches):
                X_batch, Y_batch = mnist.train.next_batch(batch_size)
                _, loss_batch = sess.run([optimizer, loss],
                                         feed_dict={X: X_batch, Y: Y_batch})
                total_loss += loss_batch
            avg_loss = total_loss / n_batches
            if debug:
                print('Average loss epoch {0}: {1}'
                      .format(i, avg_loss))
        total_time = time.time() - start_time
        if debug:
            print('Total time: {0} seconds'.format(total_time))

        # should be around 0.35 after 25 epochs
        if debug:
            print('Optimization Finished!')

        # test the model
        n_batches = int(mnist.test.num_examples / batch_size)
        total_correct_preds = 0
        for i in range(n_batches):
            X_batch, Y_batch = mnist.test.next_batch(batch_size)
            _, loss_batch, logits_batch = sess.run(
                [optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch})
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(
                tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)

        accuracy = total_correct_preds / mnist.test.num_examples
        if debug:
            print('Accuracy {0}'.
                  format(accuracy))

        writer.close()
    print('Time: {0}, Avg. Loss: {1}, Accuracy: {2}'.
          format(total_time, avg_loss, accuracy))
    return [total_time, avg_loss, accuracy]


'''
total_time, avg_loss, accuracy = analyse_logistic_regression_mnist()
print('Time: {0}, Avg. Loss: {1}, Accuracy: {2}'.
      format(total_time, avg_loss, accuracy))
'''

batch_sizes = [128, 256]

batch_result = [analyse_logistic_regression_mnist(
    batch_size=x) for x in batch_sizes]

print(batch_result)

'''
Time: 10.780983924865723, Avg. Loss: 0.3779737023271133, Accuracy: 0.9041
[[13.36782193183899, 0.33670586840811867, 0.91220000000000001], [10.780983924865723, 0.3779737023271133, 0.90410000000000001]]
'''


# Create a pandas dataframe
#results = [[13.36782193183899, 0.33670586840811867, 0.91220000000000001], [10.780983924865723, 0.3779737023271133, 0.90410000000000001]]
cols = ['Time', 'Avg. Loss', 'Accuracy']
idx = ['128', '256']
df = pd.DataFrame(batch_result, index=idx, columns=cols)
df
print(df)
