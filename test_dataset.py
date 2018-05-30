print('hello test dataset')
import tensorflow as tf
import numpy as np
sess = tf.Session()

# v = tf.random_uniform([1,6])

# iterator = tf.data.Dataset.from_tensor_slices(v).make_initializable_iterator()
# next = iterator.get_next()

# with tf.Session() as sess:
#     print(sess.run(v))
    # sess.run(iterator.initializer)
    # for _ in range(4):
    #     print(sess.run(next))
#

# dataset = tf.data.Dataset.from_tensor_slices(v)
# print(dataset.output_types)
# print(dataset.output_shapes)
#
# it = dataset.make_one_shot_iterator() #error
# next = it.get_next()
#
# for i in range(5):
#     print(sess.run(next))

# dataset = tf.data.Dataset.range(100)
# it = dataset.make_one_shot_iterator()
# next = it.get_next()
#
#
# for i in range(100):
#     v = sess.run(next)
#     print(v)


# v = [
#     [1,2],
#     [3,4],
#     [5,6]
# ]
#
# dataset = tf.data.Dataset.from_tensor_slices(v)
# next = dataset.make_one_shot_iterator().get_next()
#
# for i in range(3):
#     print(sess.run(next))
# # [1 2]
# # [3 4]
# # [5 6]

# v = tf.constant([[1,2], [2,3], [3,4], [4,5]], dtype=tf.float32)
#
# dataset = tf.data.Dataset.from_tensor_slices(v)
# next = dataset.make_one_shot_iterator().get_next()
#
# for i in range(3):
#     print(sess.run(next))

# index = tf.placeholder(tf.int64)
# dataset = tf.data.Dataset.range(index)
# it = dataset.make_initializable_iterator()
# next = it.get_next()
#
# sess.run(it.initializer, feed_dict={index:10})
# for i in range(10):
#     print(sess.run(next))

# training_dataset = tf.data.Dataset.range(100).map(
#     lambda x: x + tf.random_uniform([], -10, 10, tf.int64)
# )

# simple = tf.data.Dataset.range(10)
# simple2 = simple.map(lambda x : x + 100)
# next = simple2.make_one_shot_iterator().get_next()
#
# for i in range(10):
#     print(sess.run(next))


# simple = tf.data.Dataset.range(10)
# simple2 = simple.map(lambda x : x + tf.random_uniform([], -10, 10, tf.int64))
# next = simple2.make_one_shot_iterator().get_next()
#
# for i in range(10):
#     print(sess.run(next))

# dataset1 = tf.data.Dataset.range(10).map(lambda x : x + 1000)
# dataset2 = tf.data.Dataset.range(10).map(lambda x : x + tf.random_uniform([], -10, 10, tf.int64))
#
# it = tf.data.Iterator.from_structure(dataset1.output_types, dataset2.output_shapes)
# next = it.get_next()
#
# init_1 = it.make_initializer(dataset1)
# init_2 = it.make_initializer(dataset2)
#
# for _ in range(3):
#     sess.run(init_1)
#     for _ in range(10):
#         print(sess.run(next))
#
#     sess.run(init_2)
#     for _ in range(10):
#         print(sess.run(next))

# repeat_dataset = tf.data.Dataset.range(10).repeat() #无限重复
# next = repeat_dataset.make_one_shot_iterator().get_next()
#
# for _ in range(1000):
#     print(sess.run(next))

# training_dataset = tf.data.Dataset.range(100).map(
#     lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
# validation_dataset = tf.data.Dataset.range(50)
#
# # A feedable iterator is defined by a handle placeholder and its structure. We
# # could use the `output_types` and `output_shapes` properties of either
# # `training_dataset` or `validation_dataset` here, because they have
# # identical structure.
# handle = tf.placeholder(tf.string, shape=[])
# iterator = tf.data.Iterator.from_string_handle(
#     handle, training_dataset.output_types, training_dataset.output_shapes)
# next_element = iterator.get_next()
#
# # You can use feedable iterators with a variety of different kinds of iterator
# # (such as one-shot and initializable iterators).
# training_iterator = training_dataset.make_one_shot_iterator()
# validation_iterator = validation_dataset.make_initializable_iterator()
#
# # The `Iterator.string_handle()` method returns a tensor that can be evaluated
# # and used to feed the `handle` placeholder.
# training_handle = sess.run(training_iterator.string_handle())
# validation_handle = sess.run(validation_iterator.string_handle())
#
# # Loop forever, alternating between training and validation.
# while True:
#   # Run 200 steps using the training dataset. Note that the training dataset is
#   # infinite, and we resume from where we left off in the previous `while` loop
#   # iteration.
#   for _ in range(200):
#     print(sess.run(next_element, feed_dict={handle: training_handle}))
#
#   # Run one pass over the validation dataset.
#   sess.run(validation_iterator.initializer)
#   for _ in range(50):
#     print(sess.run(next_element, feed_dict={handle: validation_handle}))

# dataset = tf.data.Dataset.range(5)
# it = dataset.make_initializable_iterator()
# next = it.get_next()
#
# result = tf.add(next, next)
#
# sess.run(it.initializer)
# print(sess.run(result))  # ==> "0"
# print(sess.run(result))  # ==> "2"
# print(sess.run(result))  # ==> "4"
# print(sess.run(result))  # ==> "6"
# print(sess.run(result))  # ==> "8"
#
# try:
#     sess.run(result)
# except tf.errors.OutOfRangeError:
#     print('end of dataset')

# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
# dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100]))) #第一个是4行
# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
#
# iterator = dataset3.make_initializable_iterator()
#
# sess.run(iterator.initializer)
# next1, (next2, next3) = iterator.get_next()
# next = iterator.get_next()
#
# for _ in range(4):
#     print(sess.run(next))



# filenames = tf.placeholder(tf.string, shape=[None])
# dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.map(...)  # Parse the record into tensors.
# dataset = dataset.repeat()  # Repeat the input indefinitely.
# dataset = dataset.batch(32)
# iterator = dataset.make_initializable_iterator()

# dateset1 = tf.data.Dataset.range(100)
# dateset2 = tf.data.Dataset.range(0, -100, -1)
# dateset3 = tf.data.Dataset.zip((dateset1, dateset2))
# batch_dataset = dateset3.batch(4)
#
# next = batch_dataset.make_one_shot_iterator().get_next()
#
# for _ in range(5):
#     print(sess.run(next))


# dataset = tf.data.Dataset.range(100)
# dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
# dataset = dataset.padded_batch(8, padded_shapes=[None])
#
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
# print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
#                                #      [5, 5, 5, 5, 5, 0, 0],
#                                #      [6, 6, 6, 6, 6, 6, 0],
#                                #      [7, 7, 7, 7, 7, 7, 7]]
# print(sess.run(next_element))
# print(sess.run(next_element))
# print(sess.run(next_element))

# dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5,2)))
# iterator = dataset.make_one_shot_iterator()
# one_element = iterator.get_next()
# with tf.Session() as sess:
#     try:
#         while True:
#             print(sess.run(one_element))
#     except tf.errors.OutOfRangeError:
#         print("end!")

#
# {'a': 1.0, 'b': array([0.08992267, 0.28729095])}
# {'a': 2.0, 'b': array([0.61135189, 0.26666089])}
# {'a': 3.0, 'b': array([0.47938285, 0.11555829])}
# {'a': 4.0, 'b': array([0.48996376, 0.8598215 ])}
# {'a': 5.0, 'b': array([0.89530159, 0.69854528])}

dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    }
)

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

