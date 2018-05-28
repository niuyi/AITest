print('hello test api')

import tensorflow as tf
import numpy as np

sess = tf.Session()

# a = tf.constant(3.0, dtype=tf.float32)
# b = tf.constant(4.0, dtype=tf.float32)
# total = a + b
# print(a)
# print(b)
# print(total)
#
# # writer = tf.summary.FileWriter('.')
# # writer.add_graph(tf.get_default_graph())
#
# # with tf.Session() as sess:
# #     writer = tf.summary.FileWriter('D:/Code/github/tensorflow-models-master/models-master/samples/core/get_started/')
# #     writer.add_graph(tf.get_default_graph())
#
# print('done!')
#
# sess = tf.Session()
# print(sess.run(total))
# print(sess.run(a))
# print(sess.run({'ab': (a,b), 'total':total}))
#
# vec = tf.random_uniform(shape=(1,3))
# print(vec.get_shape())
# print(sess.run(vec))
# #
# vec2 = tf.random_uniform(shape=(3,))
# out1 = vec2 + 1
# print(sess.run(vec2))
# print(sess.run(out1))
#
# x = tf.placeholder(tf.float32)
# print(sess.run(x, feed_dict={x:3}))

# y = tf.placeholder(tf.float32)
# z = x + y
#
# print(sess.run(z, feed_dict={x:3, y:4}))
# print(sess.run(z, feed_dict={x:[1,3], y:[2,4]}))
#
# my_data=[
#     [1,2],
#     [3,4],
#     [4,5],
#     [5,6]
# ]
#
# slices = tf.data.Dataset.from_tensor_slices(my_data)
# print(slices)
# next_item = slices.make_one_shot_iterator().get_next()
#
# while True:
#     try:
#         print(sess.run(next_item))
#     except tf.errors.OutOfRangeError:
#         break

# x = tf.placeholder(tf.float32, shape=[None, 3])
# linear_model = tf.layers.Dense(units=1)
# y = linear_model(x)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
#
# print(sess.run(y, {x:[[1,2,3], [4,5,6],[7,8,9]]}))
#
# writer = tf.summary.FileWriter('./events')
# writer.add_graph(tf.get_default_graph())

# department_column = tf.feature_column.categorical_column_with_vocabulary_list(
#     key= 'department',
#     vocabulary_list = ['sports', 'gardening']
# )
#
# department_column = tf.feature_column.indicator_column(department_column)
#
# columns = [
#     tf.feature_column.numeric_column('sales'),
#     department_column
# ]
#
# features = {
#     'sales': [5, 10, 8, 9],
#     'department' : ['sports', 'sports', 'gardening', 'gardening']
# }
#
# # features = {
# #     'sales': [
# #         [5],
# #         [10],
# #         [8],
# #         [9]
# #     ],
# #
# #     'department' : ['sports', 'sports', 'gardening', 'gardening']
# # }
#
# inputs = tf.feature_column.input_layer(features, columns)
#
# var_init = tf.global_variables_initializer()
# table_init = tf.tables_initializer()
# sess = tf.Session()
# sess.run((var_init, table_init))
#
# print(sess.run(inputs))

# x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
# y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
#
# linear_model = tf.layers.Dense(units=1)
# y_pred = linear_model(x)
#
#
# loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
#
#
# # loss = tf.losses.mean_squared_error(labels=[1,2], predictions=[3,100])
# # print('loss', sess.run(loss)) #4804 对应点误差的平方和的均值 = ((3-1)(3-1) + (100-2)(100-2))/2
#
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# for i in range(1000):
#     sess.run(train)
#     # _,loss_value = sess.run((train, loss))
#     # print(loss_value)
#
# print(sess.run(y_pred))

#
# my_int = tf.Variable('my_int', tf.int16)
# my_int = tf.constant(100, tf.int16)
#
#
# print(sess.run(my_int))
#
# my_str = tf.Variable('my_str', tf.string)
# my_str = tf.constant('hello', tf.string)
# print(sess.run(my_str))
#
# print(my_str.get_shape().ndims) #0
# print(tf.rank(tf.Variable([1,2,3], tf.int32)))
#
# mymat = tf.Variable([[7], [11]], tf.int16)
# print(tf.rank(mymat))
#
# my_image = tf.zeros([10, 299, 299, 3])
# sess.run(my_image)
# print(tf.rank(my_image))
# # print(my_image.get_shape().ndims)

#
# t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
# print(sess.run(tf.rank(t)))


# t1 = tf.constant([1, 2, 3])
# print(t1[1])
# print(sess.run(t1[1])) #2
#
# t2 = tf.constant([
#     [1,2],
#     [3,4],
#     [5,6]
# ])
# #
# # print(sess.run(t2[1])) #[3 4]
# # print(sess.run(t2[1,1])) #4
# # print(sess.run(t2[:,1])) #[2 4 6]
# # print(t2)
# # print(t2.shape)
# # print(sess.run(t2))
# # t3 = t2*t2
# # print(t3.eval(session=sess))
#
#
# # temp = tf.Print(t2,[t2])
# # result = temp + 1
# # print(result.eval(session=sess))
#
# my_v = tf.get_variable('my_v', [3,4], initializer=tf.zeros_initializer)
# my_v2 = tf.get_variable('my_v2', initializer=my_v.initialized_value() + 1)
#
# init = tf.global_variables_initializer()#初始化所有变量
# sess.run(init)
#
# # sess.run(my_v.initializer) #直接初始化变量自己
# print(my_v.eval(session=sess))
# print(my_v2.eval(session=sess))
#
# # tf.add_to_collection("my_collection_name", my_v)
# # print(tf.get_collection("my_collection_name"))


# v = tf.get_variable("v", shape=(2,3), initializer=tf.zeros_initializer())
# assignment = v.assign_add([[1,2,3],[4,5,6]])
# sess.run(tf.global_variables_initializer())
# print(sess.run(assignment))  # or assignment.op.run(), or assignment.eval()

