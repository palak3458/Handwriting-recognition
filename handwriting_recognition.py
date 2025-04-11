import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


# Load MNIST dataset
'''mnist = tf.keras.datasets.mnist #accessing the mnist dataset of of Tenserflow's keras API. MNIST is a collection of 28x28 grayscale images of handwritten digits (0-9)
mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values (0-255 → 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(28, 28)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Save the model
model.save('handwritten.keras')

model = tf.keras.models.load_model('handwritten.keras')
loss, accuracy = model. evaluate(x_test, y_test)
print(loss)
print(accuracy)'''
# Load your trained model
model = tf.keras.models.load_model('handwritten.keras')

# Load your handwritten digit image (must be 28x28 or will be resized)
img = load_img('digit3.jpg', color_mode='grayscale', target_size=(28, 28))

# Convert to numpy array
img_array = img_to_array(img)  # shape might be (28, 28, 1)

# Normalize pixel values to [0, 1]
img_array = img_array / 255.0

# OPTIONAL: If your image is black digit on white background, invert it
# img_array = 1.0 - img_array

# If the array has shape (28, 28, 1), squeeze the channel dimension
if img_array.shape[-1] == 1:
    img_array = np.squeeze(img_array)  # now shape is (28, 28)

# Reshape to match model input: (batch_size, height, width)
img_array = img_array.reshape(1, 28, 28)

# Visualize the digit
plt.imshow(img_array[0], cmap='gray')
plt.title("Input Digit")
plt.axis('off')
plt.show()

# Predict the digit
predictions = model.predict(img_array)
predicted_digit = np.argmax(predictions)

# Output prediction
print("🔢 Predicted digit is:", predicted_digit)