import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Load data
data_dir = 'mnist/'
log_dir = 'logcnn1/'
mnist = read_data_sets(data_dir)

# Convert images into 28x28 (they are downloaded as 1x784)
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

# Convert labels into one-hot encoded vectors
train_labels = mnist.train.labels
test_labels = mnist.test.labels

# Set model parameters
generations = 10

batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels) + 1
num_channels = 1  # greyscale = 1 channel
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2  # NxN window for 1st max pool layer
max_pool_size2 = 2  # NxN window for 2nd max pool layer
fully_connected_size1 = 100


with tf.name_scope("graph"):
    with tf.name_scope("variables"):
        x_input_shape = (batch_size, image_width, image_height, num_channels)
        x_input = tf.placeholder(
            tf.float32, shape=x_input_shape, name="train_x")
        y_target = tf.placeholder(
            tf.int32, shape=(batch_size),  name="train_y")

        eval_input_shape = (evaluation_size, image_width,
                            image_height, num_channels)
        eval_input = tf.placeholder(
            tf.float32, shape=eval_input_shape,  name="test_x")
        eval_target = tf.placeholder(
            tf.int32, shape=(evaluation_size), name="test_y")

        # Convolutional layer variables
        conv1_weight = tf.Variable(tf.truncated_normal(
            [4, 4, num_channels, conv1_features],
            stddev=0.1, dtype=tf.float32), name="conv1_W")
        conv1_bias = tf.Variable(tf.zeros(
            [conv1_features],
            dtype=tf.float32), name="conv1_B")
        conv2_weight = tf.Variable(tf.truncated_normal(
            [4, 4, conv1_features, conv2_features],
            stddev=0.1, dtype=tf.float32), name="conv2_W")
        conv2_bias = tf.Variable(tf.zeros(
            [conv2_features], dtype=tf.float32), name="conv2_B")

        # fully connected variables
        resulting_width = image_width // (max_pool_size1 * max_pool_size2)  # 7
        resulting_height = image_height // (max_pool_size1 * max_pool_size2) # 7
        full1_input_size = resulting_width * resulting_height * conv2_features  # 7*7*50=2450

        full1_weight = tf.Variable(tf.truncated_normal(
            [full1_input_size, fully_connected_size1],
            stddev=0.1, dtype=tf.float32), name="full1_W")
        full1_bias = tf.Variable(tf.truncated_normal(
            [fully_connected_size1],
            stddev=0.1, dtype=tf.float32), name="full1_B")
        full2_weight = tf.Variable(tf.truncated_normal(
            [fully_connected_size1, target_size],
            stddev=0.1, dtype=tf.float32), name="full2_W")
        full2_bias = tf.Variable(tf.truncated_normal(
            [target_size],
            stddev=0.1, dtype=tf.float32), name="full2_B")

    # Initialize Model Operations
    def my_conv_net(input_data, graph_name):
        # with tf.name_scope(graph_name):
        # First Conv-ReLU-MaxPool Layer
        with tf.name_scope("conv"):
            conv1 = tf.nn.conv2d(input_data, conv1_weight,
                                    strides=[1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
            max_pool1 = tf.nn.max_pool(relu1,
                                        ksize=[1, max_pool_size1,
                                                max_pool_size1, 1],
                                        strides=[1, max_pool_size1,
                                                max_pool_size1, 1],
                                        padding='SAME')
            tf.summary.histogram("weights", conv1_weight)
            tf.summary.histogram("biases", conv1_bias)

        # Second Conv-ReLU-MaxPool Layer
        with tf.name_scope("conv"):
            conv2 = tf.nn.conv2d(max_pool1, conv2_weight,
                                    strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
            max_pool2 = tf.nn.max_pool(relu2,
                                        ksize=[1, max_pool_size2,
                                                max_pool_size2, 1],
                                        strides=[1, max_pool_size2,
                                                max_pool_size2, 1],
                                        padding='SAME')
            tf.summary.histogram("weights", conv2_weight)
            tf.summary.histogram("biases", conv2_bias)

        # Transform Output into a 1xN layer for next fully connected layer
        with tf.name_scope("reshape"):
            final_conv_shape = max_pool2.get_shape().as_list()
            final_shape = final_conv_shape[1] * \
                final_conv_shape[2] * final_conv_shape[3]
            flat_output = tf.reshape(
                max_pool2, [final_conv_shape[0], final_shape])

        # First Fully Connected Layer
        with tf.name_scope("fc"):
            fully_connected1 = tf.nn.relu(
                tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
            tf.summary.histogram("weights", full1_weight)
            tf.summary.histogram("biases", full1_bias)

        # Second Fully Connected Layer
        with tf.name_scope("fc"):
            final_model_output = tf.add(
                tf.matmul(fully_connected1, full2_weight), full2_bias)
            fully_connected1 = tf.nn.relu(
                tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
            tf.summary.histogram("weights", full2_weight)
            tf.summary.histogram("biases", full2_bias)

        return final_model_output

    with tf.name_scope("train"):
        model_output = my_conv_net(x_input, "train")

        # Declare Loss Function (softmax cross entropy)
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=model_output, labels=y_target))
            tf.summary.scalar("loss", loss)

        with tf.name_scope("accuracy"):
            model_output = tf.Print(model_output, [model_output, model_output.get_shape(), model_output[0]], "model_output = ", first_n=5, summarize=10)
            b_pred = tf.argmax(model_output, 1)
            # Debug
            b_pred = tf.Print(b_pred, [b_pred, b_pred.get_shape(), b_pred[0]], "b_pred = ")
            correct_prediction = tf.equal(tf.cast(b_pred, tf.int32), y_target)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        # Create an optimizer
        with tf.name_scope("optimizer"):
            my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
            train_step = my_optimizer.minimize(loss)

        # Create a prediction function
        with tf.name_scope("optimizer"):
            prediction = tf.nn.softmax(model_output)
    
    
    with tf.name_scope("test"):
        test_model_output = my_conv_net(eval_input, "test")

        # Create a prediction function
        with tf.name_scope("optimizer"):
            test_prediction = tf.nn.softmax(test_model_output)

    with tf.name_scope("global_ops"):
        # Initialize Variables
        init = tf.global_variables_initializer()
        summ = tf.summary.merge_all()

    # Create accuracy function
    def get_accuracy(logits, targets):
        batch_predictions = np.argmax(logits, axis=1)
        num_correct = np.sum(np.equal(batch_predictions, targets))
        ret_val = 100. * num_correct / batch_predictions.shape[0]
        return(ret_val)

sess.run(init)
writer = tf.summary.FileWriter(log_dir)
writer.add_graph(sess.graph)


# Start training loop
train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
    rand_index = np.random.choice(len(train_xdata),
                                  size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input: rand_x, y_target: rand_y}

    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds, s = sess.run(
        [loss, prediction, summ], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)

    if (i + 1) % eval_every == 0:
        # Write summaries
        writer.add_summary(s, i)
        eval_index = np.random.choice(len(test_xdata),
                                      size=evaluation_size)
        eval_x = test_xdata[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels[eval_index]
        test_dict = {eval_input: eval_x, eval_target: eval_y}
        test_preds = sess.run(test_prediction, feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds, eval_y)

        # Record and print results
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i + 1), temp_train_loss,
                        temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.
              format(*acc_and_loss))

writer.flush()
writer.close()
sess.close()
