print('test api2')
import tensorflow as tf

# c_0 = tf.constant(1, name="c") #Tensor("c:0", shape=(), dtype=int32)
# with tf.name_scope("outer"):
#     c_1 = tf.constant(1, name="c")  #Tensor("outer/c:0", shape=(), dtype=int32)
#     with tf.name_scope("inner"):
#         c_2 = tf.constant(3, name="c") #Tensor("outer/inner/c:0", shape=(), dtype=int32)
#
#     with tf.name_scope("inner"):
#         c_3 = tf.constant(3, name="c") #Tensor("outer/inner_1/c:0", shape=(), dtype=int32)
#
# c_4 = tf.constant(0, name="c") #Tensor("c_1:0", shape=(), dtype=int32)
#
# print(c_0) #Tensor("c:0", shape=(), dtype=int32)
# print(c_1) #Tensor("outer/c:0", shape=(), dtype=int32)
# print(c_2)
# print(c_3)
# print(c_4)

# Tensor("c:0", shape=(), dtype=int32)
# Tensor("outer/c:0", shape=(), dtype=int32)
# Tensor("outer/inner/c:0", shape=(), dtype=int32)
# Tensor("outer/inner_1/c:0", shape=(), dtype=int32)
# Tensor("c_1:0", shape=(), dtype=int32)
#
# 请注意，tf.Tensor 对象以输出张量的 tf.Operation 明确命名。张量名称的形式为 "<OP_NAME>:<i>"，其中：
#
# "<OP_NAME>" 是生成该张量的指令的名称。
# "<i>" 是一个整数，它表示该张量在指令的输出中的索引。
# x = tf.constant([
#     [9.0, 1.0]
# ])
#
# output = tf.nn.softmax(x)
#
# with tf.Session() as sess:
#     print(sess.run(x))
#     print(sess.run(output))


#
# x = tf.constant([
#     [37.0, -23.0],
#     [1.0, 4.0]
# ])
#
# w = tf.Variable(tf.random_uniform([2,2]))
# y = tf.matmul(x, w)
# output = tf.nn.softmax(y)
#
# init_op = w.initializer
#
# with tf.Session() as sess:
#     sess.run(init_op)
#
#     print(sess.run(x))
#     print(sess.run(w))
#     print(sess.run(y))
#
#     print(sess.run(output))
#
#
# # Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)
#
init_op = tf.global_variables_initializer()
#
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(inc_v1))
    print(sess.run(dec_v2))
    print(inc_v1.op.run())
#
#
test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
reshape = tf.data.Dataset.from_tensor_slices(tf.reshape(test, [3, 3]))
next = reshape.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    print(sess.run(next))
    print(sess.run(next))

