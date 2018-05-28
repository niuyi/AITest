import tensorflow as tf
from tensorflow.python.framework import dtypes
#
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = (sess.run(op))
    print(res.shape)
#
# x = tf.constant([[1., 2., 3.],
#                  [4., 5., 6.]])
#
# x = tf.reshape(x, [1, 2, 3, 1])  # give a shape accepted by tf.nn.max_pool
#
# valid_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
# same_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#
# valid_pad.get_shape() == [1, 1, 1, 1]  # valid_pad is [5.]
# same_pad.get_shape() == [1, 1, 2, 1]  # same_pad is  [5., 6.]

# input2 = tf.Variable(tf.random_normal([1,8,8,1]))
# op = tf.layers.conv2d(
#     inputs=input2,
#     filters=32,
#     kernel_size=[5, 5],
#     padding="same",
#     activation=tf.nn.relu)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(input2))
#
#     print(sess.run(op))
#     print(op.shape)

hot = tf.one_hot(indices=[1, 2, 3], depth=4)
print(hot)

with tf.Session() as sess:
    print(sess.run(hot))