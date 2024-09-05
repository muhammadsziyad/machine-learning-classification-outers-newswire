# models/dense_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_dense_model(input_shape=(200,), num_classes=46, num_words=10000):
    model = models.Sequential([
        layers.Embedding(input_dim=num_words, output_dim=128, input_length=input_shape[0]),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(46, activation='softmax')  # Output layer for 46 classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Sparse categorical cross-entropy for multi-class classification
                  metrics=['accuracy'])
    return model
