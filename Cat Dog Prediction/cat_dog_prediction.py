from keras import layers
from keras.models import Sequential

##########################
# Create a Sequential model
##########################
model = Sequential()

# Add a Conv2D layer with 32 filters, a 3x3 kernel, and ReLU activation function
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

##########################
# Compile the model
##########################
from keras import optimizers

model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=["acc"],
)

##########################
# Data Generator
##########################
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import seaborn as sns

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# paths to the train and validation directories
train_path = "./data/catdog/train"
validation_path = "./data/catdog/validation"

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(150, 150), batch_size=20, class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_path, target_size=(150, 150), batch_size=20, class_mode="binary"
)

##########################
# Fit the model
##########################
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50,
)

##########################
# Plot the results
##########################
import matplotlib.pyplot as plt

epochs = np.arange(1, 21)

plt.plot(epochs, history.history["loss"], label="Training Loss")
plt.plot(epochs, history.history["val_loss"], label="Validation Loss")

plt.plot(epochs, history.history["acc"], label="Training Accuracy")
plt.plot(epochs, history.history["val_acc"], label="Validation Accuracy")

plt.title("Training Loss & Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss & Accuracy")

plt.legend(loc="upper right")
plt.show()

"""
we have an overfitting problem. The training loss and accuracy are improving, but the validation loss and accuracy are not.

Possible solutions:
1. Creating new model
2. Data Augmentation
"""

##########################
# 1. Creating new model
##########################

second_model = Sequential()

# Add a Conv2D layer with 32 filters, a 3x3 kernel, and ReLU activation function
second_model.add(
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3))
)
second_model.add(layers.MaxPooling2D((2, 2)))

second_model.add(layers.Conv2D(64, (3, 3), activation="relu"))
second_model.add(layers.MaxPooling2D((2, 2)))

second_model.add(layers.Conv2D(128, (3, 3), activation="relu"))
second_model.add(layers.MaxPooling2D((2, 2)))

second_model.add(layers.Conv2D(128, (3, 3), activation="relu"))
second_model.add(layers.MaxPooling2D((2, 2)))

second_model.add(layers.Flatten())
second_model.add(layers.Dropout(0.5))

second_model.add(layers.Dense(512, activation="relu"))
second_model.add(layers.Dense(1, activation="sigmoid"))

second_model.summary()

##########################
# Compile the new model
##########################
second_model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=["acc"],
)

##########################
# 2. Data Augmentation
##########################

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)


train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(150, 150), batch_size=16, class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_path, target_size=(150, 150), batch_size=16, class_mode="binary"
)

##########################
# Fit the new model
##########################

second_history = second_model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
)
print(f"info: Done!\n {second_history.history}")

##########################
# Plot the results
##########################
epochs = np.arange(1, 101)

plt.plot(epochs, second_history.history["loss"], label="Training Loss")
plt.plot(epochs, second_history.history["val_loss"], label="Validation Loss")

plt.plot(epochs, second_history.history["acc"], label="Training Accuracy")
plt.plot(epochs, second_history.history["val_acc"], label="Validation Accuracy")

plt.title("Training Loss & Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss & Accuracy")

plt.legend(loc="upper right")
plt.show()


##########################
# Save the models
##########################
model.save("./models/cat_dog_model.h5")
second_model.save("./models/cat_dog_model_v2.h5")

##########################
# Predictions
##########################

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import random

# learning labels
labels = train_generator.class_indices
print(f"info: Labels: {labels}")

random_list = ["cats", "dogs"]
random_choice = random.choice(random_list)

# Load the model
model_path = "./models/cat_dog_model_v2.h5"
img_path = f"./data/catdog/test/{random_choice}/{np.random.randint(1500,2000)}.jpg"

prediction_model = load_model(model_path)
test_img = load_img(img_path, target_size=(150, 150))

# Convert the image to an array
test_img_array = img_to_array(test_img)
test_img_array = np.expand_dims(test_img_array, axis=0)

prediction = prediction_model.predict(test_img_array)
print(f"info: Prediction: {prediction}")

if prediction < 0.5:
    random_choice = "Cat"
else:
    random_choice = "Dog"


# putting labels on the image
import cv2 as cv

test_img = cv.imread(img_path)
cv.putText(
    test_img,
    f"Prediction: {random_choice}",
    (10, 30),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
    cv.LINE_AA,
)
cv.imshow("Prediction", test_img)
cv.waitKey(0)
cv.destroyAllWindows()
