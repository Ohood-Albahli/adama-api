import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121


data_dir = r"C:\Users\ohood\Downloads\Adama-all-main\Adama-all-main\src\model\Eczema and  Psoriasis Merged\test"
csv_path = os.path.join(data_dir, "_classes.csv")


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, 224, 224)
    return image / 255.0, label

