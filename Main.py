# %% Importing the necessary libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.metrics import SparseCategoricalCrossentropy
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# %% Getting the dataset loaded:

train = pd.read_csv(r"/kaggle/input/emnist/emnist-digits-train.csv")
test = pd.read_csv(r"/kaggle/input/emnist/emnist-digits-test.csv")

# %% Let's separate out the labels and pic:

print(train.shape)
print(train.describe())

y1 = np.array(train.iloc[:, 0].values)
x1 = np.array(train.iloc[:, 1:].values)

y2 = np.array(test.iloc[:, 0].values)
x2 = np.array(test.iloc[:, 1:].values)

# %% Let's Visualize the data:

pic = 10
print(y1[pic])
plt.imshow(np.array(x1[pic]).reshape((28, 28)).T)
plt.show()

# %% Change the pictures into grey scaling:

plt.plot(np.array(x1[1]))
plt.show()

x1 = x1/255.0
print(x1.max())

x2 = x2/255.0
print(x2.max())

plt.plot(np.array(x1[1]))
plt.show()

# %% Reshape the images:

batch_number_train = train.shape[0]
batch_number_test = test.shape[0]
img_height = 28
img_width = 28
channels = 1

x1 = x1.reshape(batch_number_train, img_height, img_width, 1)
x2 = x2.reshape(batch_number_test, img_height, img_width, 1)

# %% Encoding the label values:

number_of_classes = 10
y1 = to_categorical(y1, number_of_classes)
y2 = to_categorical(y2, number_of_classes)

# %% Splitting the training and testing datasets:

X_train = x1
y_train = y1
X_test = x2
y_test = y2
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# %% Improvements in the design of models:

ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=3,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.0001)
MCP = ModelCheckpoint('Best_Model.keras',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')

# %% Compile the model with Sparse Categorical Cross-Entropy loss:

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32, callbacks=[ES, RLP, MCP])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Test Loss: {loss:.4f}')

# %% Final Result of the Accuracy:

sn.lineplot(x = np.arange(1, len(history.history['accuracy'])+1), y = history.history['accuracy'],label='Accuracy')
sn.lineplot(x = np.arange(1, len(history.history['val_accuracy'])+1), y = history.history['val_accuracy'], label='Val_Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# %%

