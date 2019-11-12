import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sklearn.metrics
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from data_preprocessing import open_npy_data, compute_class_weight

CLASSES = ['Minimalism', 'Romanticism', 'Rococo', 'Post_Impressionism', 'Art_Nouveau_Modern', 'Renaissance', 'Pointillism', 'Realism', 'Ukiyo_e', 'Symbolism', 'Baroque', 'Cubism', 'Abstract', 'Pop_Art', 'Impressionism', 'Expressionism', 'Color_Field_Painting']
data_dir ="processed_dataset_arrays" #Where the npy arrays are saved
# 3500 or less images per class, 180x180x3 pixels


train_data, train_labels, test_data, test_labels = open_npy_data(data_dir) # loading the data
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

#normalizing the data
train_data = np.array(train_data, dtype=np.float32)
train_data /=255
test_data = np.array(test_data,dtype=np.float32)
test_data/=255
print(train_data[0])

class_weight_dict = compute_class_weight(train_labels)

#trainsforming the labels array in one-hot encoded ones
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

#loading the pre-trained model
base_model = VGG16(include_top=False,
                   weights='imagenet',
                   input_shape=train_data[0].shape)

#freezing the first 4 blocks
set_trainable = False
for layer in base_model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = keras.models.Sequential()
model.add(base_model)
model.add(keras.layers.Flatten(name="block6_flatten"))
model.add(keras.layers.Dense(512, activation="relu", name="block6_dense1"))
model.add(keras.layers.Dense(256, activation="relu", name="block6_dense2"))
model.add(keras.layers.Dense(17, activation="sigmoid"))

opt = keras.optimizers.Adam(decay=1e-2 / 10)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['acc'])
print(model.summary())

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                              patience=5, min_lr=0.0001)

history = model.fit(train_data,
                    train_labels,
                    epochs=10,
                    batch_size=512,
                    class_weight=class_weight_dict,
                    validation_data=(test_data, test_labels),
                    callbacks=[reduce_lr])

model.save("fine_tuned_VGG16_180x180.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# plotting loss and accuracy during training
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predictions = model.predict(test_data)
test_labels = np.argmax(test_labels, axis=1)

#print confusion matrix to have a better ideia of the nn predictions
confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predictions)
print(confusion_matrix)
