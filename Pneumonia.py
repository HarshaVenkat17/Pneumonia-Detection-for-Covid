#Load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Creating the CNN model
classifier = Sequential()

# 3 pairs of alternate Colvolutional and Pooling layers
classifier.add(Conv2D(32, (3, 3), input_shape = (400,400,1),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3),input_shape = (400,400,1),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), input_shape = (400,400,1),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening layer
classifier.add(Flatten())

# Fully connected Dense layers
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile the model
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# Data Generation

train_gen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2, vertical_flip = True)
val_gen = ImageDataGenerator(rescale = 1./255)
test_gen = ImageDataGenerator(rescale = 1./255)

train_set = train_gen.flow_from_directory('../input/pneumonia-xray-images/train',
                                          target_size = (400,400),batch_size = 8,
                                          class_mode = 'binary',color_mode = 'grayscale')

val_set = val_gen.flow_from_directory('../input/pneumonia-xray-images/val',
                                     target_size = (400,400),batch_size = 8,
                                     class_mode = 'binary',color_mode = 'grayscale')

test_set = test_gen.flow_from_directory('../input/pneumonia-xray-images/test',
                                        target_size = (400,400),batch_size = 1,class_mode = None,
                                        color_mode = 'grayscale',shuffle=False)

# Fitting the model
results = classifier.fit(train_set, steps_per_epoch = len(train_set),
                                   epochs = 50,validation_data = val_set,validation_steps = len(val_set))

#Plotting Accuracy and Loss graphs
plt.plot(results.history['accuracy']) 
plt.plot(results.history['val_accuracy']) 
plt.title('Model Accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

plt.plot(results.history['loss']) 
plt.plot(results.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show() 
 

# Predictions on test set
predictions = classifier.predict_generator(test_set)
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

# Model Performance
cm = pd.DataFrame(data=confusion_matrix(test_set.classes, predictions, labels=[0, 1]),
                  index=["Actual Normal", "Actual Pneumonia"],
                  columns=["Predicted Normal", "Predicted Pneumonia"])
print(cm)
diagonal=cm["Predicted Normal"]["Actual Normal"]+cm["Predicted Pneumonia"]["Actual Pneumonia"]
total=diagonal+ cm["Predicted Normal"]["Actual Pneumonia"]+cm["Predicted Pneumonia"]["Actual Normal"]
print("Accuracy=", diagonal/total)
