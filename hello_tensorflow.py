import sys
import tensorflow as tf

print(sys.version)
print(sys.version_info)

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
