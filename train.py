# train.py

import tensorflow as tf
from data.reuters_data import load_data
from models.dense_model import build_dense_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load Reuters data
(x_train, y_train), (x_test, y_test) = load_data()

# Build the dense model
model = build_dense_model()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Train the model
history = model.fit(x_train, y_train, epochs=20,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, reduce_lr],
                    batch_size=64)

# Save the trained model
model.save('dense_reuters_model.h5')

# Save the training history
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
