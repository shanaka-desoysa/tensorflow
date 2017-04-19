import sys
import tensorflow as tf

print(sys.version)
print(sys.version_info)

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


# Normalize vector
x = [-1.0,0.0,1.0]
x_normalize = tf.nn.l2_normalize(x, dim = 0)
print(sess.run(x_normalize))
