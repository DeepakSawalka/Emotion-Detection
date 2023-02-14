import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.applications import VGG16, InceptionResNetV2
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
import tensorflow as tf
#print(tf.__version__)
label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
model = tf.keras.models.load_model('model_optimal.h5')
# Load the previously saved weights
model.load_weights('model_weights.h5')
path="F:\\mini_project_ml\\static\\image"
for img1 in os.listdir(path):
    imgpath = os.path.join(path,img1)
    img = image.load_img(imgpath,target_size = (48,48),color_mode = "grayscale")
    img = np.array(img)
    plt.imshow(img)
    #print(img.shape) #prints (48,48) that is the shape of our image
    img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
    img = img.reshape(1,48,48,1)
    result = model.predict(img)
    print("result:",result)
    result = list(result[0])
    print("Results:",result)
    #print(result)
    img_index = result.index(max(result))
    print(label_dict[img_index])
    plt.show()
