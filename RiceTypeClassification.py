#!/usr/bin/env python
# coding: utf-8

# # Business Understanding
# 
# The dataset used is "Rice Image Dataset" by Murat Koklu, it is found on Kaggle at this link: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/code?datasetId=2049052&sortBy=voteCount
# 
# This report aims to demonstrate the effectiveness of using CNN architecture which can be further used on more complex data to assist industries such as the agriculture business through deep learning architecture in efficient identification of various product types.

# In[ ]:


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import pandas as pd
import random
import cv2
import os
import PIL
import pathlib

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ### Loading and Splitting Dataset

# In[ ]:


unique_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

data_dir = '/content/drive/MyDrive/Rice_Dataset/'

all_paths = []
all_labels = []

for label in unique_labels:
    for image_path in os.listdir(data_dir+label):
        img = cv2.imread(data_dir+label+'/'+image_path)
        img_resize = cv2.resize(img, dsize=(100, 100))
        # img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        all_paths.append(img_resize)
        if(label == "Arborio"):
          all_labels.append(1)
        elif(label == "Basmati"):
          all_labels.append(2)
        elif(label == "Ipsala"):
          all_labels.append(3)
        elif(label == "Jasmine"):
          all_labels.append(4)
        elif(label == "Karacadag"):
          all_labels.append(5)

# store image paths and labels
all_paths, all_labels = shuffle(all_paths, all_labels)


# ### Choice of Metric: Classification Accuracy

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(all_paths, all_labels,
                                                              test_size=0.2, random_state=42, shuffle = True,
                                                              stratify=all_labels)


# ### Train, Test, Split

# In[ ]:


fig, ax = plt.subplots(ncols=5, figsize=(20,5))
fig.suptitle('Rice Category')
ax[0].set_title('Arborio')
ax[1].set_title('Basmati')
ax[2].set_title('Ipsala')
ax[3].set_title('Jasmine')
ax[4].set_title('Karacadag')


ax[0].imshow(all_paths[100],cmap='gray')
ax[1].imshow(all_paths[1100],cmap='gray')
ax[2].imshow(all_paths[2100],cmap='gray')
ax[3].imshow(all_paths[3100],cmap='gray')
ax[4].imshow(all_paths[4100],cmap='gray')


# In[ ]:


del [all_paths, all_labels]


# In[ ]:


X_train = np.array(X_train)
X_test = np.array(X_test)


# In[ ]:


X_train = X_train/255.0 - 0.5
X_test = X_test/255.0 - 0.5


# In[ ]:


y_train_ohe = keras.utils.to_categorical(y_train,6)
y_test_ohe = keras.utils.to_categorical(y_test,6)


# In[ ]:


del [y_test,y_train ]


# ### Data Expansion in Keras

# In[ ]:


datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=5, # used, Int. Degree range for random rotations.
    width_shift_range=0.1, # used, Float (fraction of total width). Range for random horizontal shifts.
    height_shift_range=0.1, # used,  Float (fraction of total height). Range for random vertical shifts.
    shear_range=0., # Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None)

datagen.fit(X_train)

idx = 0


# In[ ]:


plt.imshow(X_train[121],cmap=plt.get_cmap('gray'))


# # Modeling

# ## Model 1: Ensemble Nets Variation 1
# 
# 1.   L2 Regularization

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimg_wh = 100\nNUM_CLASSES = 6\n\nfrom tensorflow.keras.layers import Input, average, concatenate\nfrom tensorflow.keras.models import Model\n\nnum_ensembles = 3\nl2_lambda = 0.000001\n#l1 for second variation\n#replace    kernel_regularizer=keras.regularizers.l2(l2_lambda),\n#with this:\n# kernel_regularizer=keras.regularizers.l1_l2(l1_lambda, l2_lambda)\n# l1_lambda = 0.00001\n\n\ninput_holder = Input(shape=(img_wh, img_wh, 3))\n\n# start with a conv layer\nx = Conv2D(filters=32,\n               input_shape = (img_wh,img_wh,3),\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l2(l2_lambda),\n               padding=\'same\', \n               activation=\'relu\', data_format="channels_last")(input_holder)\n\nx = Conv2D(filters=32,\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l2(l2_lambda),\n               padding=\'same\', \n               activation=\'relu\')(x)\ninput_conv = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\nbranches = []\nfor _ in range(num_ensembles):\n    \n    # start using NiN (MLPConv)\n    x = Conv2D(filters=32,\n                   input_shape = (img_wh,img_wh,3),\n                   kernel_size=(3,3),\n                   kernel_initializer=\'he_uniform\', \n                   kernel_regularizer=keras.regularizers.l2(l2_lambda),\n                   padding=\'same\', \n                   activation=\'linear\', data_format="channels_last")(input_conv)\n\n    x = Conv2D(filters=32,\n                   kernel_size=(1,1),\n                   kernel_initializer=\'he_uniform\', \n                   kernel_regularizer=keras.regularizers.l2(l2_lambda),\n                   padding=\'same\', \n                   activation=\'relu\', data_format="channels_last")(x)\n    \n    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n    \n    x = Conv2D(filters=64,\n                   input_shape = (img_wh,img_wh,3),\n                   kernel_size=(3,3),\n                   kernel_initializer=\'he_uniform\', \n                   kernel_regularizer=keras.regularizers.l2(l2_lambda),\n                   padding=\'same\', \n                   activation=\'linear\', data_format="channels_last")(x)\n\n    x = Conv2D(filters=64,\n                   kernel_size=(1,1),\n                   kernel_initializer=\'he_uniform\', \n                   kernel_regularizer=keras.regularizers.l2(l2_lambda),\n                   padding=\'same\', \n                   activation=\'linear\', data_format="channels_last")(x)\n    \n    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\n    # add one layer on flattened output\n    x = Flatten()(x)\n    x = Dropout(0.50)(x) # add some dropout for regularization after conv layers\n    x = Dense(64, \n              activation=\'relu\',\n              kernel_initializer=\'he_uniform\',\n              kernel_regularizer=keras.regularizers.l2(l2_lambda)\n            )(x)\n    \n    x = Dense(NUM_CLASSES, \n              activation=\'relu\',\n              kernel_initializer=\'he_uniform\',\n              kernel_regularizer=keras.regularizers.l2(l2_lambda)\n             )(x)\n    \n    # now add this branch onto the master list\n    branches.append(x)\n\n# that\'s it, we just need to average the results\nx = concatenate(branches)\n\nx = Dense(NUM_CLASSES, \n          activation=\'softmax\', \n          kernel_initializer=\'glorot_uniform\',\n          kernel_regularizer=keras.regularizers.l2(l2_lambda)\n         )(x)\n\n# here is the secret sauce for setting the network using the \n#   Functional API:\ncnn_ens = Model(inputs=input_holder,outputs=x)\n\ncnn_ens.summary()')


# In[ ]:


# Let's train the model 
cnn_ens.compile(loss='categorical_crossentropy', # 'categorical_crossentropy' 'mean_squared_error'
                optimizer='adam', # 'adadelta' 'rmsprop'
                metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', "# the flow method yields batches of images indefinitely, with the given transformations\ncnn1_history = cnn_ens.fit_generator(datagen.flow(X_train, y_train_ohe, batch_size=128), \n                      steps_per_epoch=int(len(X_train)/128), # how many generators to go through per epoch\n                      epochs=50, verbose=1,\n                      validation_data=(X_test,y_test_ohe),\n                      callbacks=[EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0001)])")


# In[ ]:


del[cnn_ens]


# In[ ]:


cnn1_history.history
plt.plot(cnn1_history.history["accuracy"])
plt.plot(cnn1_history.history["val_accuracy"])
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('CNN Ensemble 1 Accuracy')
ax = plt.gca()
ax.legend(['training accuracy', 'test accuracy'])
plt.show()
print()
#-----------------------------------------
plt.plot(cnn1_history.history["loss"])
plt.plot(cnn1_history.history["val_loss"])
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('CNN Ensemble 1 Loss')
ax = plt.gca()
ax.legend(['training loss', 'test loss'])
plt.show()


# ## Model 2: Ensemble Nets Variation 2
# 
# 1.   L1 Regularization
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom tensorflow.keras.layers import Input, average, concatenate\nfrom tensorflow.keras.models import Model\n\nimg_wh = 100\nNUM_CLASSES = 6\n\nnum_ensembles = 3\nl2_lambda = 0.000001\n#l1 for second variation\n#replace    kernel_regularizer=keras.regularizers.l2(l2_lambda),\n#with this:\n# kernel_regularizer=keras.regularizers.l1_l2(l1_lambda, l2_lambda)\nl1_lambda = 0.00001\n\n\ninput_holder = Input(shape=(img_wh, img_wh, 3))\n\n# start with a conv layer\nx = Conv2D(filters=32,\n               input_shape = (img_wh,img_wh,3),\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l1(l1_lambda),\n               padding=\'same\', \n               activation=\'relu\', data_format="channels_last")(input_holder)\n\nx = Conv2D(filters=32,\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l1(l1_lambda),\n               padding=\'same\', \n               activation=\'relu\')(x)\ninput_conv = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\nbranches = []\nfor _ in range(num_ensembles):\n    \n    # start using NiN (MLPConv)\n    x = Conv2D(filters=32,\n                   input_shape = (img_wh,img_wh,3),\n                   kernel_size=(3,3),\n                   kernel_initializer=\'he_uniform\', \n                   kernel_regularizer=keras.regularizers.l1(l1_lambda),\n                   padding=\'same\', \n                   activation=\'linear\', data_format="channels_last")(input_conv)\n\n    x = Conv2D(filters=32,\n                   kernel_size=(1,1),\n                   kernel_initializer=\'he_uniform\', \n                   kernel_regularizer=keras.regularizers.l1(l1_lambda),\n                   padding=\'same\', \n                   activation=\'relu\', data_format="channels_last")(x)\n    \n    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n    \n    x = Conv2D(filters=64,\n                   input_shape = (img_wh,img_wh,3),\n                   kernel_size=(3,3),\n                   kernel_initializer=\'he_uniform\', \n                   kernel_regularizer=keras.regularizers.l1(l1_lambda),\n                   padding=\'same\', \n                   activation=\'linear\', data_format="channels_last")(x)\n\n    x = Conv2D(filters=64,\n                   kernel_size=(1,1),\n                   kernel_initializer=\'he_uniform\', \n                   kernel_regularizer=keras.regularizers.l1(l1_lambda),\n                   padding=\'same\', \n                   activation=\'linear\', data_format="channels_last")(x)\n    \n    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\n    # add one layer on flattened output\n    x = Flatten()(x)\n    x = Dropout(0.50)(x) # add some dropout for regularization after conv layers\n    x = Dense(64, \n              activation=\'relu\',\n              kernel_initializer=\'he_uniform\',\n              kernel_regularizer=keras.regularizers.l1(l1_lambda)\n            )(x)\n    \n    x = Dense(NUM_CLASSES, \n              activation=\'relu\',\n              kernel_initializer=\'he_uniform\',\n              kernel_regularizer=keras.regularizers.l1(l1_lambda)\n             )(x)\n    \n    # now add this branch onto the master list\n    branches.append(x)\n\n# that\'s it, we just need to average the results\nx = concatenate(branches)\n\nx = Dense(NUM_CLASSES, \n          activation=\'softmax\', \n          kernel_initializer=\'glorot_uniform\',\n          kernel_regularizer=keras.regularizers.l1(l1_lambda)\n         )(x)\n\n# here is the secret sauce for setting the network using the \n#   Functional API:\ncnn_ens2 = Model(inputs=input_holder,outputs=x)\n\ncnn_ens2.summary()')


# In[ ]:


# Let's train the model 
cnn_ens2.compile(loss='categorical_crossentropy', # 'categorical_crossentropy' 'mean_squared_error'
                optimizer='adam', # 'adadelta' 'rmsprop'
                metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', "# the flow method yields batches of images indefinitely, with the given transformations\ncnn2_history = cnn_ens2.fit_generator(datagen.flow(X_train, y_train_ohe, batch_size=128), \n                      steps_per_epoch=int(len(X_train)/128), # how many generators to go through per epoch\n                      epochs=50, verbose=1,\n                      validation_data=(X_test,y_test_ohe),\n                      callbacks=[EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0001)])")


# In[ ]:


cnn2_history.history
plt.plot(cnn2_history.history["accuracy"])
plt.plot(cnn2_history.history["val_accuracy"])
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('CNN Ensemble 2 Accuracy')
ax = plt.gca()
ax.legend(['training accuracy', 'test accuracy'])
plt.show()
print()
#-----------------------------------------
plt.plot(cnn2_history.history["loss"])
plt.plot(cnn2_history.history["val_loss"])
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('CNN Ensemble 2 Loss')
ax = plt.gca()
ax.legend(['training loss', 'test loss'])
plt.show()


# ## Model 3: Resnet Variation 1

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# now lets use the LeNet architecture with batch norm\n# We will also use ReLU where approriate and drop out \nfrom tensorflow.keras.layers import Add, Input\nfrom tensorflow.keras.layers import average, concatenate\nfrom tensorflow.keras.models import Model\n\nimg_wh = 100\n\ninput_holder = Input(shape=(img_wh, img_wh, 3))\n\n# start with a conv layer\nx = Conv2D(filters=32,\n               input_shape = (img_wh,img_wh,3),\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l2(l2_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(input_holder)\n\nx = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\nx = Conv2D(filters=32,\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l2(l2_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(x)\n\nx_split = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\nx = Conv2D(filters=64,\n               kernel_size=(1,1),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l2(l2_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(x_split)\n\nx = Conv2D(filters=64,\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l2(l2_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(x)\n\nx = Conv2D(filters=32,\n               kernel_size=(1,1),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l2(l2_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(x)\n\n# now add back in the split layer, x_split (residual added in)\nx = Add()([x, x_split])\nx = Activation("relu")(x)\n\nx = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\nx = Flatten()(x)\nx = Dropout(0.25)(x)\nx = Dense(256)(x)\nx = Activation("relu")(x)\nx = Dropout(0.5)(x)\nx = Dense(NUM_CLASSES)(x)\nx = Activation(\'softmax\')(x)\n\nresnet = Model(inputs=input_holder,outputs=x)\n\nresnet.summary()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "resnet.compile(loss='categorical_crossentropy', # 'categorical_crossentropy' 'mean_squared_error'\n                optimizer='adam', # 'adadelta' 'rmsprop'\n                metrics=['accuracy'])\n\nresnet1_history = resnet.fit(X_train, y_train_ohe, batch_size=128, \n                      epochs=50, verbose=1,\n                      validation_data=(X_test,y_test_ohe),\n                      callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]\n                     )")


# In[ ]:


del[resnet]


# In[ ]:


resnet1_history.history
plt.plot(resnet1_history.history["accuracy"])
plt.plot(resnet1_history.history["val_accuracy"]) 
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('Resnet 1 Accuracy')
ax = plt.gca()
ax.legend(['training accuracy', 'test accuracy'])
plt.show()
print()

#-----------------------------------------
plt.plot(resnet1_history.history["loss"])
plt.plot(resnet1_history.history["val_loss"])
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('Resnet 1 Loss')
ax = plt.gca()
ax.legend(['training loss', 'test loss'])
plt.show()


# ## Model 4: Resnet Variation 2

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# now lets use the LeNet architecture with batch norm\n# We will also use ReLU where approriate and drop out \nfrom tensorflow.keras.layers import Add, Input\nfrom tensorflow.keras.layers import average, concatenate\nfrom tensorflow.keras.models import Model\n\nimg_wh = 100\n\ninput_holder = Input(shape=(img_wh, img_wh, 3))\n\n# start with a conv layer\nx = Conv2D(filters=32,\n               input_shape = (img_wh,img_wh,3),\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l1(l1_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(input_holder)\n\nx = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\nx = Conv2D(filters=32,\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l1(l1_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(x)\n\nx_split = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\nx = Conv2D(filters=64,\n               kernel_size=(1,1),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l1(l1_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(x_split)\n\nx = Conv2D(filters=64,\n               kernel_size=(3,3),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l1(l1_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(x)\n\nx = Conv2D(filters=32,\n               kernel_size=(1,1),\n               kernel_initializer=\'he_uniform\', \n               kernel_regularizer=keras.regularizers.l1(l1_lambda),\n               padding=\'same\', \n               activation=\'relu\', \n               data_format="channels_last")(x)\n\n# now add back in the split layer, x_split (residual added in)\nx = Add()([x, x_split])\nx = Activation("relu")(x)\n\nx = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)\n\nx = Flatten()(x)\nx = Dropout(0.25)(x)\nx = Dense(256)(x)\nx = Activation("relu")(x)\nx = Dropout(0.5)(x)\nx = Dense(NUM_CLASSES)(x)\nx = Activation(\'softmax\')(x)\n\nresnet = Model(inputs=input_holder,outputs=x)\n\nresnet.summary()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "resnet.compile(loss='categorical_crossentropy', # 'categorical_crossentropy' 'mean_squared_error'\n                optimizer='adam', # 'adadelta' 'rmsprop'\n                metrics=['accuracy'])\n\nresnet2_history = resnet.fit(X_train, y_train_ohe, batch_size=128, \n                      epochs=50, verbose=1,\n                      validation_data=(X_test,y_test_ohe),\n                      callbacks=[EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0001)]\n                     )")


# In[ ]:


del[resnet]


# In[ ]:


resnet2_history.history
plt.plot(resnet2_history.history["accuracy"])
plt.plot(resnet2_history.history["val_accuracy"]) 
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Resnet 2 Accuracy')
ax = plt.gca()
ax.legend(['training accuracy', 'test accuracy'])
plt.show()
print()

#-----------------------------------------
plt.plot(resnet2_history.history["loss"])
plt.plot(resnet2_history.history["val_loss"])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Resnet 2 Loss')
ax = plt.gca()
ax.legend(['training loss', 'test loss'])
plt.show()


# ### Best Fit Model

# ### MLP Adaptation

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Flatten())\nmodel.add(Dense(input_dim = 1000, units = 2500, activation='relu'))\nmodel.add(Dense(500, activation='relu'))\nmodel.add(Dense(250, activation='relu'))\nmodel.add(Dense(6, activation='softmax'))\n\n# Configure the model and start training\nmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\nmlphistory = model.fit(X_train, y_train_ohe, batch_size=128, \n                      epochs=50, verbose=1,\n                      validation_data=(X_test,y_test_ohe),\n                      callbacks=[EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0001)]\n                     )")


# In[ ]:


mlphistory.history
plt.plot(mlphistory.history["accuracy"])
plt.plot(mlphistory.history["val_accuracy"]) 
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('MLP Accuracy') 
ax = plt.gca()
ax.legend(['training accuracy', 'test accuracy'])
plt.show()
print()

#-----------------------------------------
plt.plot(mlphistory.history["loss"])
plt.plot(mlphistory.history["val_loss"])
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('MLP Loss')
ax = plt.gca()
ax.legend(['training loss', 'test loss'])
plt.show()


# ### MLP vs CNN Comparisons
# 
# Reference: https://www.linkedin.com/pulse/mlp-vs-cnn-rnn-deep-learning-machine-model-momen-negm/
# 

# ## Transfer Learning with Resnet

# In[ ]:


# connect new layers to the output
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# In[ ]:


get_ipython().run_cell_magic('time', '', "resTransferModel = Sequential()\npreTrans = ResNet50(include_top = False, classes = 6, weights='imagenet')\nresTransferModel.add(Flatten())\nresTransferModel.add(Dense(500, activation='relu'))\nresTransferModel.add(Dense(250, activation='relu'))\nresTransferModel.add(Dense(6, activation='softmax'))\n\n# Configure the resTransferModel and start training\nresTransferModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\nresTransferHistory = resTransferModel.fit(X_train, y_train_ohe, batch_size=128, \n                      epochs=20, verbose=1,\n                      validation_data=(X_test,y_test_ohe),\n                      callbacks=[EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0001)] # 13 epochs last time\n                     )")


# In[ ]:


resTransferHistory.history
plt.plot(resTransferHistory.history["accuracy"])
plt.plot(resTransferHistory.history["val_accuracy"]) 
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('Transfer Learning Resnet Accuracy')
ax = plt.gca()
ax.legend(['training accuracy', 'test accuracy'])
plt.show()
print()

#-----------------------------------------
plt.plot(resTransferHistory.history["loss"])
plt.plot(resTransferHistory.history["val_loss"])
plt.xlabel('epoch') 
plt.ylabel('accuracy') 
plt.title('Transfer Learning Resnet  Loss')
ax = plt.gca()
ax.legend(['training loss', 'test loss'])
plt.show()


# ### Best CNN vs Transfer Learning

# In[ ]:


get_ipython().run_cell_magic('shell', '', 'jupyter nbconvert --to html /content/Lab6.ipynb')

