#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import numpy as np


LEARNING_RATE = 1e-4
IMAGE_NAME= "image"

tf.logging.set_verbosity(tf.logging.INFO)

def create_model(data_format):
  """Model to recognize digits in the MNIST dataset.

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But uses the tf.keras API.

  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats

  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  model = create_model(params['data_format'])
  image = features
  if isinstance(image, dict):
    image = features[IMAGE_NAME]

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    # If we are running multi-GPU, we need to wrap the optimizer.
    if params.get('multi_gpu'):
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    logits = model(image, training=True)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(LEARNING_RATE, 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(tf.nn.softmax(logits), name='softmax_tensor_adjoshi')
    tf.identity(tf.argmax(logits), name='classes_adjoshi')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy':
                tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)),
        })


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  data_format = None
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir="/tmp/mnist_convnet_model",
      params={
          'data_format': data_format,
          'multi_gpu': False
      })

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"train_accuracy": "softmax_tensor_adjoshi","class":"classes_adjoshi"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=1)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={IMAGE_NAME: train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=30,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={IMAGE_NAME: eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()