
import math
import tensorflow as tf

#print(sys.version)
#print(sys.version_info)
a = tf.placeholder(tf.float32,
                       shape=[None],
                       name="input_placeholder_a")
x = tf.placeholder(tf.float32,
                       shape=[None],
                       name="input_placeholder_x")
b = tf.sqrt(tf.nn.moments(a, axes=[0])[1], name="product_b")

c = tf.norm(a)
d = a / c

normed_a = tf.nn.l2_normalize(a, dim=0)
normed_x = tf.nn.l2_normalize(x, dim=0)
#normed_x_T = tf.transpose([normed_x])
cosine_similarity = tf.matmul([normed_a], tf.transpose([normed_x]))

cov = tf.c


hello = tf.constant('Hello, TensorFlow!')



sess = tf.Session()
print(sess.run(hello))
#[[1],[2],[3],[4],[5]]
#[1,2,3,4,5]
feed_dict = {a: [1,2,3,4,5], x: [-1,-2,-3,-4,-5]}

#x = sess.run([b,c,d,e], feed_dict=feed_dict)
x = sess.run([cosine_similarity], feed_dict=feed_dict)

#print( math.sqrt(x))
print(x)

#print(c)


'''
import numpy as np

all_xs = []
all_ys = []
for i in range(5):
    # Create fake data for y = 2.x_1 + 5.x_2 + 7
    #x_1 = i % 10
    #x_2 = np.random.randint(datapoint_size / 2) % 10
    #y = actual_W1 * x_1 + actual_W2 * x_2 + actual_b
    # Create fake data for y = W.x + b where W = [2, 5], b = 7
    all_xs.append(i+1)
    all_ys.append(i+5+(i%2))

all_xs = np.array(all_xs)
all_ys = np.transpose([all_ys])

print(all_xs)
print(all_ys)
'''
