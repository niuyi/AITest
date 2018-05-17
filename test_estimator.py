import tensorflow as tf
import numpy as np
import pandas as pd
import iris_data
from tensorflow.python.framework import dtypes

print('hello test!')

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

def load_data():
    path = 'D:/Code/github/tensorflow-models-master/models-master/data/iris_training.csv'
    train = pd.read_csv(path, names=['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species'], header=0)

    train_x = train
    train_y = train.pop('Species')

    test = pd.read_csv('D:/Code/github/tensorflow-models-master/models-master/data/iris_test.csv',
                       names=['SepalLength', 'SepalWidth',
                              'PetalLength', 'PetalWidth', 'Species'], header=0
                       )
    test_x = test
    test_y = test.pop('Species')

    return (train_x, train_y),(test_x, test_y)


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)

    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)

    return dataset

COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']

FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]


def parse_line(line):
    print('line', dir(line))
    fields = tf.decode_csv(line, record_defaults=FIELD_DEFAULTS)
    features = dict(zip(COLUMNS, fields))
    print('features', features)
    for k,v in features.items():
        print('f', k, v)
    label = features.pop('label')
    print('label', label)
    return features, label

def parse_csv():
    train_path, test_path = iris_data.maybe_download()
    ds = tf.data.TextLineDataset(train_path).skip(1)
    print('parse_csv')
    print(ds)
    ds = ds.map(parse_line)

def parse_csv2():
    train_path, test_path = iris_data.maybe_download()
    iris_data.csv_input_fn(train_path, 1000)

parse_csv()

#
my_features_columns = [tf.feature_column.numeric_column(key='SepalLength'),
                       tf.feature_column.numeric_column(key='SepalWidth'),
                       tf.feature_column.numeric_column(key='PetalLength'),
                       tf.feature_column.numeric_column(key='PetalWidth')]

#分桶列
# f1 = tf.feature_column.numeric_column(key='SepalLength')
# bf1 = tf.feature_column.bucketized_column(source_column=f1, boundaries=[5,6,7])
#
# f2 = tf.feature_column.numeric_column(key='SepalWidth')
# bf2 = tf.feature_column.bucketized_column(source_column=f2, boundaries=[2.5,3,3.5])
#
# f3 = tf.feature_column.numeric_column(key='PetalLength')
# bf3 = tf.feature_column.bucketized_column(source_column=f3, boundaries=[2, 3, 4])
#
# f4 = tf.feature_column.numeric_column(key='PetalWidth')
# bf4 = tf.feature_column.bucketized_column(source_column=f4, boundaries=[0.5, 1, 1.5])
#
# my_features_columns = [bf1, bf2, bf3, bf4]

#经过哈希处理的列
# my_features_columns = [tf.feature_column.categorical_column_with_hash_bucket(key='SepalLength', hash_bucket_size = 100),
#                        tf.feature_column.categorical_column_with_hash_bucket(key='SepalWidth', hash_bucket_size = 100),
#                        tf.feature_column.categorical_column_with_hash_bucket(key='PetalLength', hash_bucket_size = 100),
#                        tf.feature_column.categorical_column_with_hash_bucket(key='PetalWidth', hash_bucket_size = 100)]

# my_checkpointing_config = tf.estimator.RunConfig(
#     save_checkpoints_secs=20,
#     keep_checkpoint_max=10,
#     model_dir='D:/Code/github/tensorflow-models-master/models-master/check_points'
# )

my_classifier = tf.estimator.DNNClassifier(
    feature_columns=my_features_columns,
    hidden_units=[10, 10],
    n_classes=3
)

(train_x, train_y),(test_x, test_y) = load_data()


# train_x = {
#     'SepalLength': np.array([6.4, 5, 4.9, 4.9, 5.7]),
#     'SepalWidth': np.array([2.8, 2.3, 2.5, 3.1, 3.8]),
#     'PetalLength': np.array([5.6, 3.3, 4.5, 1.5, 1.7]),
#     'PetalWidth': np.array([2.2, 1, 1.7, 0.1, 0.3])
# }
#
# train_y = np.array([2, 1, 2, 0, 0])
# train_y = [2, 1, 2, 0, 0]

my_classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y, 100) ,
    steps=100
)

print('end train')
result = my_classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, 100)
)
print('end eval')
print('result', result)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**result))


expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = my_classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x, labels=None,batch_size=100)
)

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')



for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(expected[class_id],
                          100 * probability, expec))

