import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class FashionMNIST(keras.Model):
    def __init__(self, num_classes=10):
        super(FashionMNIST, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.dense(x)

# Load the Fashion MNIST dataset
(_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values
x_test = x_test.astype("float32") / 255.0

# Reshape images to (28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# Load the trained model (assuming you ran the training code before)
num_classes = 10
model = FashionMNISTCNN(num_classes=num_classes)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# Get predictions for a few images
num_predictions = 2
random_indices = np.random.choice(x_test.shape[0], num_predictions, replace=False)
selected_images = x_test[random_indices]
true_labels = y_test[random_indices]

predictions = model.predict(selected_images)

# Get the predicted class labels
predicted_labels = np.argmax(predictions, axis=1)

# Define class names for visualization
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize the predictions
plt.figure(figsize=(10, 5))
for i in range(num_predictions):
    plt.subplot(1, num_predictions, i + 1)
    plt.imshow(selected_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[true_labels[i]]}\nPredicted: {class_names[predicted_labels[i]]} ({np.max(predictions[i]):.2f})")
    plt.axis('off')
plt.tight_layout()
plt.show()
