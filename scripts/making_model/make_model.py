# -*- coding: utf-8 -*-
"""Lerning_example

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W0jsA7vp6q3zya8INpF_k9hXsHOXB-_j
"""

!pip install -q tflite-model-maker

import os

 import numpy as np

 import tensorflow as tf
 assert tf.__version__.startswith('2')

 from tflite_model_maker import model_spec
 from tflite_model_maker import image_classifier
 from tflite_model_maker.config import ExportFormat
 from tflite_model_maker.config import QuantizationConfig
 from tflite_model_maker.image_classifier import DataLoader

 import matplotlib.pyplot as plt

# load example pictures
data = DataLoader.from_folder("/content/drive/MyDrive/flower_photos")

# train & test
train_data, test_data = data.split(0.9)
model = image_classifier.create(train_data)
loss, accuracy = model.evaluate(test_data)
print(loss)
print(accuracy)

# export trained model
model.export(export_dir='/content/')


# TODO - import trained model
interpreter = tf.lite.Interpreter(model_path="/content/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)