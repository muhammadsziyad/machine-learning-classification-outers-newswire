# data/reuters_data.py

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(num_words=10000, max_len=200):
    # Load Reuters dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=num_words)

    # Pad sequences (each article to have the same length)
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    return (x_train, y_train), (x_test, y_test)
