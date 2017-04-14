# pylint: disable=invalid-name

# Softmax example in TF using the classical Iris dataset
# Download iris.data from https://archive.ics.uci.edu/ml/datasets/Iris

import os
import tensorflow as tf

DATA_FILE = "data/IrisDataSet.csv"
LOG_FILE = "logs/p3_iris"


def combine_inputs(X):
    with tf.name_scope("combine_inputs"):
        return tf.matmul(X, W) + b


def inference(X):
    with tf.name_scope("inference"):
        return tf.nn.softmax(combine_inputs(X))


def loss(X, Y):
    with tf.name_scope("loss"):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=combine_inputs(X),
                labels=Y))


def read_csv(batch_size, file_name, record_defaults):
    with tf.name_scope("read_csv"):
        filename_queue = tf.train.string_input_producer(
            [os.path.dirname(__file__) + "/" + file_name])

        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)

        # decode_csv will convert a Tensor from type string (the text line) in
        # a tuple of tensor columns with the specified defaults, which also
        # sets the data type for each column
        decoded = tf.decode_csv(
            value, record_defaults=record_defaults, name="decode_csv")

        # batch actually reads the file and loads "batch_size" rows in a single
        # tensor
        return tf.train.shuffle_batch(decoded,
                                      batch_size=batch_size,
                                      capacity=batch_size * 50,
                                      min_after_dequeue=batch_size,
                                      name="shuffle_batch")


def inputs():
    with tf.name_scope("inputs"):
        sepal_length, sepal_width, petal_length, petal_width, label =\
            read_csv(100, DATA_FILE, [[0.0], [0.0], [0.0], [0.0], [""]])

        # convert class names to a 0 based class index.
        label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
            tf.equal(label, ["Iris-setosa"]),
            tf.equal(label, ["Iris-versicolor"]),
            tf.equal(label, ["Iris-virginica"])
        ])), 0), name="label")

        # Pack all the features that we care about in a single matrix;
        # We then transpose to have a matrix with one example per row and one
        # feature per column.
        features = tf.transpose(tf.stack(
            [sepal_length, sepal_width, petal_length, petal_width]), name="features")

        return features, label_number


def train(total_loss):
    with tf.name_scope("train"):
        learning_rate = 0.01
        return tf.train.GradientDescentOptimizer(learning_rate, name="GradientDescent").minimize(total_loss)


def evaluate(sess, X, Y):
    with tf.name_scope("evaluate"):
        predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
        print("Evaluation: ", sess.run(tf.reduce_mean(
            tf.cast(tf.equal(predicted, Y), tf.float32))))


# Explicitly create a Graph object
graph = tf.Graph()

with graph.as_default():
    with tf.name_scope("weights_and_bias"):
        # this time weights form a matrix, not a column vector, one "weight
        # vector" per class.
        W = tf.Variable(tf.zeros([4, 3]), name="weights")
        # so do the biases, one per class.
        b = tf.Variable(tf.zeros([3], name="bias"))

    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    with tf.name_scope("summaries"):
        # Creates summary for output node
        # Scalar summary for loss
        tf.summary.scalar("loss", total_loss)

    # Global Variables and Operations
    with tf.name_scope("global_ops"):
        # Initialization Op
        init = tf.global_variables_initializer()
        # Collect all summary Ops in graph
        merged_summaries = tf.summary.merge_all()

# Launch the graph in a session, setup boilerplate
with tf.Session(graph=graph) as sess:
    # Open a SummaryWriter to save summaries
    writer = tf.summary.FileWriter(LOG_FILE, graph)

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets
        # decremented thru training steps
        if step % 10 == 0:
            loss_val, summary_str = sess.run([total_loss, merged_summaries])
            writer.add_summary(summary_str, step)
            if step % 100 == 0:
                print("loss: ", loss_val)

    evaluate(sess, X, Y)

    # Writes the summaries to disk
    writer.flush()

    # Flushes the summaries to disk and closes the SummaryWriter
    writer.close()

    coord.request_stop()
    coord.join(threads)
    sess.close()
