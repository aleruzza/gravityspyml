#imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.backend import reverse
from tensorflow import keras
import tensorflow as tf
import numpy as np

#VGG16 D
def vgg16d():

    #define inputs
    inputs = Input(shape=(60,71,3))

    #first block
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='b1_c1')(inputs)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='b1_c2')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p')(x)

    #second block
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='b2_c1')(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu', name='b2_c2')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p')(x)

    #third block
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='b3_c1')(x)
    x = Conv2D(256, (3,3), padding='same', activation='relu', name='b3_c2')(x)
    x = Conv2D(256, (3,3), padding='same', activation='relu', name='b3_c3')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b3_p')(x)

    #forth block
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='b4_c1')(x)
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='b4_c2')(x)
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='b4_c3')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b4_p')(x)

    #fifth block
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='b5_c1')(x)
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='b5_c2')(x)
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='b5_c3')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b5_p')(x)

    #dense layers
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='FC1')(x)
    x = Dense(4096, activation='relu', name='FC2')(x)
    x = Dense(22, activation='softmax', name='Out')(x)

    modelvgg = Model(inputs, x, name="VGG16d")

    return modelvgg


def doubleConvCNN(dropout=False):

    name="doubleConvDroputCNN" if dropout else "doubleConvCNN"

    #define inputs
    inputs = Input(shape=(60, 71, 3))

    #normalization
    #x = LayerNormalization(axis=[2])(inputs)

    #first block
    x = Conv2D(8, (3,3), activation='relu', padding='same', name='b1_c1')(inputs)
    x = Conv2D(8, (3,3), activation='relu', padding='same', name='b1_c2')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p')(x)

    #second block
    x = Conv2D(16, (3,3), activation='relu', padding='same', name='b2_c1')(x)
    x = Conv2D(16, (3,3), padding='same', activation='relu', name='b2_c2')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p')(x)

    #third block
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='b3_c1')(x)
    x = Conv2D(32, (3,3), padding='same', activation='relu', name='b3_c2')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b3_p')(x)

    #dense layers
    x = Flatten(name='flatten')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='FC1')(x)
    x = Dense(64, activation='relu', name='FC2')(x)
    x = Dense(22, activation='softmax', name='Out')(x)

    simplecnn = Model(inputs, x, name=name)

    return simplecnn

def singleConvCNN(dropout=False):

    name="singleConvDroputCNN" if dropout else "singleConvCNN"

    #define inputs
    inputs = Input(shape=(60, 71, 3))

    #normalization
    #x = LayerNormalization(axis=[2])(inputs)

    #first block
    x = Conv2D(8, (3,3), activation='relu', padding='same', name='b1_c1')(inputs)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b1_p', padding='same')(x)

    #second block
    x = Conv2D(16, (3,3), activation='relu', padding='same', name='b2_c1')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='b2_p', padding='same')(x)

    #third block
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='b3_c1')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same',name='b3_p')(x)

    #dense layers
    x = Flatten(name='flatten')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='FC1')(x)
    x = Dense(64, activation='relu', name='FC2')(x)
    x = Dense(22, activation='softmax', name='Out')(x)

    simplecnn = Model(inputs, x, name=name)

    return simplecnn


def LeNet5(dropout=False):

    name="LeNet5_Dropout" if dropout else "LeNet5"

    #define inputs
    inputs = Input(shape=(60,71,3), name='In')

    #convolution
    x = Conv2D(6, (5,5), activation='tanh', strides=(1,1), name='C1')(inputs)

    #avg pooling
    x = AveragePooling2D((2,2), strides=2, name='S2')(x)

    #convolution
    x = Conv2D(16, (5,5), activation="tanh", strides=(1,1), name='C3')(x)

    #avg pooling
    x = AveragePooling2D((2,2), strides=2, name='S4')(x)

    #convolution
    x = Conv2D(120, (1,1), activation="tanh", name='C5')(x)

    #flatten
    x = Flatten()(x)

    #dropout
    if(dropout):
        x = Dropout(0.5)(x)

    #fully connected layers
    x = Dense(84, activation='tanh', name='F6')(x)
    x = Dense(22, activation='softmax', name='Out')(x)

    lenet5 = Model(inputs, x, name=name)
    
    return lenet5