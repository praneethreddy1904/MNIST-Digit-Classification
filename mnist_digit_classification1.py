"""
Handwritten Digit Recognition Using Neural Networks (MNIST Dataset)
Cybersecurity/AI Internship Project - Naviotech Solution Pvt Ltd
Author: Praneeth Reddy
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# %%
# ---------------------------
# Phase 1: Data Collection
# ---------------------------
print("Phase 1: Loading MNIST dataset...")

# %%
import gzip # Not needed when using keras.datasets.mnist.load_data()

# %%
def load_images(path): # Not needed when using keras.datasets.mnist.load_data()
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

# %%
def load_labels(path): # Not needed when using keras.datasets.mnist.load_data()
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# %%
import os, gzip, urllib.request
os.makedirs('rawdata', exist_ok=True)
BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
for fname in FILES:
    fpath = os.path.join('rawdata', fname)
    if not os.path.exists(fpath):
        urllib.request.urlretrieve(BASE_URL + fname, fpath)
def _load_images(path):
    with gzip.open(path, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
def _load_labels(path):
    with gzip.open(path, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)
x_train = _load_images('rawdata/train-images-idx3-ubyte.gz')
y_train = _load_labels('rawdata/train-labels-idx1-ubyte.gz')
x_test = _load_images('rawdata/t10k-images-idx3-ubyte.gz')
y_test = _load_labels('rawdata/t10k-labels-idx1-ubyte.gz')

print(f"Training samples: {x_train.shape[0]}")
print(f"Testing samples: {x_test.shape[0]}")
print(f"Image dimensions: {x_train.shape[1]}x{x_train.shape[2]}")

# %%
# Create a directory to save images and model if it doesn't exist
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')
    print("Created 'screenshots' directory.")

# %%
# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.suptitle("Sample MNIST Digit Images")
plt.tight_layout()

# Ensure the directory exists before saving
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

plt.savefig('screenshots/01_sample_digits.png', dpi=120)
plt.close()
print("Saved: sample digit images")

# %%
# ---------------------------
# Phase 2: Data Preprocessing
# ---------------------------
print("\nPhase 2: Preprocessing data...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
print("Normalization and reshaping complete.")

# %%
# ---------------------------
# Phase 3: Model Development
# ---------------------------
print("\nPhase 3: Building Neural Network model...")
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
with open('screenshots/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# %%
# ---------------------------
# Phase 4: Model Training
# ---------------------------
print("\nPhase 4: Training model...")
history = model.fit(
    x_train, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# %%
# Plot accuracy and loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot Loss
axes[1].plot(history.history['loss'], label='Train Loss', marker='o', color='red')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='o', color='orange')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('screenshots/02_accuracy_loss_graph.png', dpi=120)
plt.close()
print("Saved: accuracy/loss graphs")

# %%
history = model.fit(
    x_train, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test_cat)
)
axes[1].plot(history.history['loss'], label='Train Loss', marker='o', color='red')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='o', color='orange')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('screenshots/02_accuracy_loss_graph.png', dpi=120)
plt.close()
print("Saved: accuracy/loss graphs")

# %%
# ---------------------------
# Phase 5: Testing and Evaluation
# ---------------------------
print("\nPhase 5: Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# %%
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
report = classification_report(y_test, y_pred)
print(report)
with open('screenshots/classification_report.txt', 'w') as f:
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\nTest Loss: {test_loss:.4f}\n\n")
    f.write(report)

# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - MNIST Digit Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('screenshots/03_confusion_matrix.png', dpi=120)
plt.close()
print("Saved: confusion matrix")

# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - MNIST Digit Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
print("Displayed: confusion matrix")

# %%
import os
import matplotlib.pyplot as plt

# Create folder if it doesn't exist
os.makedirs("screenshots", exist_ok=True)

# Train the model
history = model.fit(
    x_train,
    y_train_cat,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test_cat)
)

# Create figure with two graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy Graph
axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o', color='blue')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o', color='green')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss Graph
axes[1].plot(history.history['loss'], label='Train Loss', marker='o', color='red')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='o', color='orange')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Save graph
plt.tight_layout()
plt.savefig("screenshots/02_accuracy_loss_graph.png", dpi=120)
plt.show()

print("Saved: accuracy/loss graphs")

# %%
import matplotlib.pyplot as plt

confusion_matrix_path = 'screenshots/03_confusion_matrix.png'
img = plt.imread(confusion_matrix_path)

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title('Confusion Matrix - MNIST Digit Classification (from file)')
plt.axis('off') # Turn off axis labels and ticks
plt.show()

print(f"Displayed: Confusion Matrix from {confusion_matrix_path}")

# %%
# ---------------------------
# Phase 6: Prediction
# ---------------------------
print("\nPhase 6: Sample Predictions...")
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]} | True: {y_test[i]}")
    ax.axis('off')
plt.suptitle("Model Predictions on Test Samples")
plt.tight_layout()
plt.savefig('screenshots/04_predictions.png', dpi=120)
plt.close()
print("Saved: prediction outputs")

# %%
model.save('mnist_digit_model.h5')
print("\nModel saved as mnist_digit_model.h5")
print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%")
