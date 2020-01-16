import tensorflow as tf
import os as os
import matplotlib.pyplot as plt

try:
  print(tf.__version__) == "2.1.0"
except Exception:
  pass

def getdata():
    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

    print("Local copy of the dataset file: {}".format(train_dataset_fp))
    # column order in CSV file
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    feature_names = column_names[:-1]
    label_name = column_names[-1]

    print("Features: {}".format(feature_names))
    print("Label: {}".format(label_name))
    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    batch_size = 32

    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1)
    features, labels = next(iter(train_dataset))

    plt.scatter(features['petal_length'],
                features['sepal_length'],
                c=labels,
                cmap='viridis')

    plt.xlabel("Petal length")
    plt.ylabel("Sepal length")
    plt.show()

    train_dataset = train_dataset.map(pack_features_vector)

    features, labels = next(iter(train_dataset))

    print(features[:5])

    return features, labels, train_dataset

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

if __name__ == '__main__':
    getdata()
