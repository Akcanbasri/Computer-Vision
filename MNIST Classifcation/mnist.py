# Setting the environment variable to address the OpenMP runtime conflict
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Importing TensorFlow and Keras modules after setting the environment variable
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display the shapes of the datasets
print(x_train.shape, y_train.shape)
print("**************************")
print(x_test.shape, y_test.shape)

# Display one of the images
import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap="gray")
plt.show()

# Reshaping the data from 3D to 4D for compatibility with Keras
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape, x_test.shape)

# Normalizing the data to facilitate learning
import numpy as np

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# Building the convolutional neural network model
model = Sequential()
model.add(Conv2D(28, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

# Compiling the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

# Training the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluating the model
test_loss, test_acc = model.evaluate(x_test, y_test)

# Graphical representation of the training history
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.legend(loc="lower right")
plt.show()


# save the model
model.save("mnist.h5")

# load the model and predict
from keras.models import load_model

model = load_model("mnist.h5")

i = np.random.randint(0, x_test.shape[0])

prediction = model.predict(x_test[i].reshape(1, 28, 28, 1))
print("Predicted digit:", np.argmax(prediction))

plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
plt.show()

