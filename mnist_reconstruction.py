# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 00:07:33 2018

@author: aleja
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

# https://blog.keras.io/building-autoencoders-in-keras.html

# In[1]:

#################################################################
#################################################################
# SIMPLEST POSSIBLE AUTOENCODED
#################################################################
#################################################################

#Size our encoded representations
encoding_dim = 32

############################################
# MODEL
############################################
#Input
input_img = Input(shape = (784,))

#Encode
encoded = Dense(encoding_dim, activation='relu')(input_img)

#Decode
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded) 

############################################
# ENCODER MODEL
############################################

encoder = Model(input_img, encoded) 

############################################
# DECODER MODEL
############################################

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

############################################

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# In[2]:

############################################
# DATA 
############################################

(x_train, _), (x_test, _) = mnist.load_data()

# In[3]:

# The dataset have images with 28x28 pixels
# Normalize all values between 0 and 1

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the 28x28 images into vectors of size 784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print('X_train shape: ', x_train.shape)
print('X_test shape: ', x_test.shape)

# In[4]:

# Train
 
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# In[5]:

# Predict

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#decoded_imgs = autoencoder.predict(x_test)

# In[6]:

# Plot

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.savefig('./Result_mnist/1-simplest_autoencoder_MEAN_' + str(decoded_imgs.mean())[:4]+'.jpg')
plt.show()

print('MEAN RESULT PREDICT SIMPLEST AUTOENCODER', decoded_imgs.mean())

# In[7]:

#################################################################
#################################################################
# SPARSITY CONSTRAING AUTOENCODED
#################################################################
#################################################################    
    
############################################
# MODEL
############################################

encoding_dim = 32

input_img = Input(shape=(784,))
# add a Dense layer with a L1 activity regularizer
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

#############################################
## ENCODER MODEL
#############################################
#
#encoder = Model(input_img, encoded) 
#
#############################################
## DECODER MODEL
#############################################
#
#encoded_input = Input(shape=(encoding_dim,))
#
## retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
#
## create the decoder model
#decoder = Model(encoded_input, decoder_layer(encoded_input))

# In[8]:

# Train
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# In[9]:

# Predict

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#decoded_imgs = autoencoder.predict(x_test)

# In[10]:

# Plot

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('./Result_mnist/2-spartsity_constraing_autoencoder_MEAN_' + str(decoded_imgs.mean())[:4]+'.jpg')
plt.show()

print('MEAN RESULT PREDICT SPARSITY CONSTRAINT', decoded_imgs.mean())

# In[11]:

#################################################################
#################################################################
# DEEP AUTOENCODED
#################################################################
#################################################################    

############################################
# MODEL
############################################

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

############################################
# ENCODER MODEL
############################################

encoder = Model(input_img, encoded) 

############################################
# DECODER MODEL
############################################

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# In[12]:

# Train
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# In[13]:

# Predict

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#decoded_imgs = autoencoder.predict(x_test)

# In[14]:

# Plot

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('./Result_mnist/3-deep_autoencoder_MEAN_' + str(decoded_imgs.mean())[:4]+'.jpg')
plt.show()

print('MEAN RESULT PREDICT SPARSITY CONSTRAINT', decoded_imgs.mean())

# In[15]:

#################################################################
#################################################################
# CONVOLUTIONAL AUTOENCODED
#################################################################
#################################################################  

############################################
# MODEL
############################################
#
#input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
#
#x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)
#
## at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(16, (3, 3), activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
#decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
#autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
#############################################
## DATA
#############################################
#
#(x_train, _), (x_test, _) = mnist.load_data()
#
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 