# pylint: disable=invalid-name

import tensorflow as tf
import numpy as np

LOG_FILE = 'logs/p2'

# Explicitly create a Graph object
graph = tf.Graph()

with graph.as_default():

    with tf.name_scope("variables"):
        # Variable to keep track of how many times the graph has been run
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Increments the above `global_step` Variable, should be run whenever
        # the graph is run
        increment_step = global_step.assign_add(1)

    a = tf.placeholder(tf.float32,
                       shape=[None],
                       name="input_a")
    b = tf.placeholder(tf.float32,
                       shape=[None],
                       name="input_b")

    # Primary transformation Operations
    with tf.name_scope("exercise_transformation"):
        # Separate input layer
        with tf.name_scope("intermediate_layer_1"):
            # Create input placeholder- takes in a Vector
            with tf.name_scope("intermediate_layer_a"):
                a_moments = tf.nn.moments(a, axes=[0], name="a_sd")
                a_norm = tf.norm(a, name="a_norm")

            with tf.name_scope("intermediate_layer_b"):
                b_moments = tf.nn.moments(b, axes=[0], name="b_sd")
                b_norm = tf.norm(b, name="b_norm")

        # Separate middle layer
        with tf.name_scope("intermediate_layer_2"):
            a_normalized = tf.nn.l2_normalize(a, dim=0, name="normalize_a")
            b_normalized = tf.nn.l2_normalize(b, dim=0, name="normalize_b")

    # Separate output layer
    with tf.name_scope("cosine_ab"):
        b_normalized_T = tf.transpose([b_normalized])
        cosine_similarity = tf.matmul([a_normalized],
                                      b_normalized_T)
    
    a_sd = tf.sqrt(a_moments[1], name="a_std_dev")
    b_sd = tf.sqrt(b_moments[1], name="b_std_dev")

    with tf.name_scope("covariance_ab"):
        a_mean = tf.cast(tf.reduce_mean(a), tf.float32)
        b_mean = tf.cast(tf.reduce_mean(b), tf.float32)
        a_delta = a - a_mean
        b_delta = b - b_mean
        covariance = tf.reduce_mean(tf.multiply(a_delta, b_delta))

    # Summary Operations
    with tf.name_scope("summaries"):
        # Creates summary for output node
        tf.summary.scalar("a_sd", a_sd)
        tf.summary.scalar("b_sd", b_sd)
        tf.summary.scalar("a_norm", a_norm)
        tf.summary.scalar("b_norm", b_norm)
        tf.summary.scalar("cosine_ab", cosine_similarity[0][0])
        tf.summary.scalar("covariance_ab", covariance)

    # Global Variables and Operations
    with tf.name_scope("global_ops"):
        # Initialization Op
        init = tf.global_variables_initializer()
        # Collect all summary Ops in graph
        merged_summaries = tf.summary.merge_all()

# Start a Session, using the explicitly created Graph
sess = tf.Session(graph=graph)

# Open a SummaryWriter to save summaries
writer = tf.summary.FileWriter(LOG_FILE, graph)

# Initialize Variables
sess.run(init)


def run_graph(input_tensor1, input_tensor2):
    """
    Helper function; runs the graph with given input tensor and saves summaries
    """
    feed_dict = {a: input_tensor1, b: input_tensor2}
    # a_sd_val, b_sd_val, a_norm_val, b_norm_val, cosine_similarity_val, covariance_val, summary, step = sess.run(
    #     [a_sd, b_sd, a_norm, b_norm, cosine_similarity, covariance, merged_summaries, increment_step], feed_dict=feed_dict)
    # print("a_sd: {0}, b_sd: {1}, a_norm: {2}, b_norm: {3}, cosine: {4}, covariance: {5}".
    #       format(a_sd_val, b_sd_val, a_norm_val, b_norm_val, cosine_similarity_val, covariance_val))
    summary, step, a_mean_val, b_mean_val, covariance_val = sess.run([merged_summaries, increment_step, a_mean, b_mean, covariance], feed_dict=feed_dict)
    writer.add_summary(summary, step)
    print("a_mean: {0}, b_mean: {1}, cov: {2}".format(a_mean_val, b_mean_val,covariance_val))


#run_graph([3.0, 5.0, 355.0, 3.0], [22.0, 111.0, 3.0, 10.0])
#run_graph([3, 1, 3, 3],[3, 1, 3, 3])

def run_graph_with_random_vectors(iterations=8, seed=1234):
    np.random.seed(seed)
    for i in range(iterations):
        v_len = np.random.randint(10, 20)
        print("Vector length: {0}".format(v_len))
        x, y = [], []
        for j in range(v_len):
            x.append(np.random.randint(1, 10))
            y.append(np.random.randint(1, 10))
        print("x: {0}".format(x))
        print("y: {0}".format(y))
        run_graph(x, y)


run_graph_with_random_vectors()


# Writes the summaries to disk
writer.flush()

# Flushes the summaries to disk and closes the SummaryWriter
writer.close()

# Close the session
sess.close()

# To start TensorBoard after running this file, execute the following command:
# $ tensorboard --logdir='./improved_graph'
