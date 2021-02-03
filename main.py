
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist


(train_images,train_lables),(test_images,test_lables) = data.load_data()

class_name = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

print(class_name[1])
print(train_lables[1])

train_images = train_images/255.0
test_images = test_images/255.0

print(train_images[7])

plt.imshow(train_images[7])
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Input layer
    keras.layers.Dense(128, activation="relu"), # Hidden layer
    keras.layers.Dense(10, activation="softmax") # Output layer
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_lables, epochs=15) # epochs means how many times we want the model to see the input data

test_loss, test_acc = model.evaluate(test_images, test_lables)
print("Tested Acc:", test_acc)
