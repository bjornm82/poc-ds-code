#!/usr/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import tensorflow as tf

try:
  print(tf.__version__) == "2.1.0"
except Exception:
  pass

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train(features, labels, train_dataset):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
  ])

  predictions = model(features)
  print(predictions[:5])

  tf.nn.softmax(predictions[:5])

  print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
  print("    Labels: {}".format(labels))
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  l = loss(model, features, labels, training=False)
  print("Loss test: {}".format(l))

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  loss_value, grads = grad(model, features, labels)

  print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                            loss_value.numpy()))

  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                            loss(model, features, labels, training=True).numpy()))

  ## Note: Rerunning this cell uses the same model variables

  # Keep results for plotting
  train_loss_results = []
  train_accuracy_results = []

  num_epochs = 201

  for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
      # Optimize the model
      loss_value, grads = grad(model, x, y)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      # Track progress
      epoch_loss_avg(loss_value)  # Add current batch loss
      # Compare predicted label to actual label
      # training=True is needed only if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      epoch_accuracy(y, model(x, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
      print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                  epoch_loss_avg.result(),
                                                                  epoch_accuracy.result()))
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

if __name__ == '__main__':
    train()