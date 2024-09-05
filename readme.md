Let's create a TensorFlow project using Keras with the Reuters newswire dataset for multi-class classification. This project will involve building a neural network to classify news articles into different categories (46 classes).

Project Structure
Here’s a typical structure for the project:

```css
tensorflow_reuters/
│
├── data/
│   └── reuters_data.py            # Script to load and preprocess Reuters data
├── models/
│   ├── dense_model.py             # Script to define and compile a neural network model
├── train.py                       # Script to train the model
├── evaluate.py                    # Script to evaluate the trained model
└── utils/
    └── plot_history.py            # Script to plot training history
```


Step 1: Load and Preprocess Reuters Data
Create a file named reuters_data.py in the data/ directory. This script will handle loading and preprocessing the Reuters dataset.

```python
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
```

Step 2: Define the Neural Network Model
We'll define a fully connected (dense) neural network for multi-class classification in the dense_model.py file inside the models/ directory.

```python
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
```

Step 3: Train the Model
Create a train.py script at the root of the project to load data, build the model, and train it.

```python
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
model.save('dense_reuters_model.keras')

# Save the training history
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
```

Step 4: Evaluate the Model
Create an evaluate.py script to evaluate the trained model on the test data.

```python
# evaluate.py

import tensorflow as tf
from data.reuters_data import load_data

# Load Reuters data
(x_train, y_train), (x_test, y_test) = load_data()

# Load the trained model
model = tf.keras.models.load_model('dense_reuters_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Step 5: Plot Training History
Use the plot_history.py script to visualize the training and validation accuracy and loss.

```python
# utils/plot_history.py

import matplotlib.pyplot as plt
import pickle

def plot_history(history_file='history.pkl'):
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    plot_history()
```

Step 6: Run the Project
Train the Model: Run the train.py script to start training the model.

```bash
python train.py
```

Evaluate the Model: After training, evaluate the model's performance using evaluate.py.

```bash
python evaluate.py
```

Plot the Training History: Visualize the training history using plot_history.py.

```bash
python utils/plot_history.py
```

Summary
This project demonstrates how to use the Reuters newswire dataset with TensorFlow and Keras to perform multi-class classification using a fully connected (dense) neural network. The model processes sequences of words from the news articles, with padding to ensure uniform length.