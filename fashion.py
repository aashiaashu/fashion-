import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from skimage.transform import resize
from skimage import data
import numpy as np
import pandas as pd
import scipy

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import resnet50, VGG16 , InceptionV3, Xception
from tensorflow.keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Dropout,MaxPooling2D,AveragePooling2D,Activation


from keras import optimizers
from keras.utils import to_categorical
from keras.models import Model
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

train_images.shape
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
resize(x, (100, 100))
def resize_all(x, shape = (48,48)):
    band_shape = x.shape
    x_resize = np.zeros(shape = (band_shape[0],shape[0],shape[1]))
    for i in range(band_shape[0]):
        x_resize[i] = resize(x[i], shape)
    return x_resize


def transform_input_vgg(x):
    x_vgg = np.array(x).reshape(-1,28,28)
    x_vgg = resize_all(x_vgg, (48,48))
    x_vgg = np.repeat(x_vgg[:, :, :, np.newaxis], 3, axis=3)
#    x_vgg = preprocess_input(x_vgg)
    return x_vgg
vgg_conv = VGG16(weights= None , include_top=False, 
                     input_shape=(48, 48, 3))
vgg_conv.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg_conv.trainable = False
def vgg16_model():
    vgg_conv = VGG16(weights= None , include_top=False, 
                     input_shape=(48, 48, 3))
    vgg_conv.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg_conv.trainable = False
    model = Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))   
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

prediction_layer=keras.layers.Dense(10, activation='softmax',name='predictions')
# global_average_layer,
#     prediction_layer

model = tf.keras.Sequential([
    # Add the vgg convolutional base model
    vgg_conv,

    # Add new layers
    Flatten(),
    Dense(1024, activation='relu'), 
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
X_train = transform_input_vgg(train_images)
X_test = transform_input_vgg(test_images)
