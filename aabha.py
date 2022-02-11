from __future__ import annotations

#@ ---------------------------------------- DRDO PROJECT ----------------------------------------
__authors__: list[str] = [
    'Aabha Malik',  'Manasdeep Singh', 'Sarthak Srivastav'
]

__authors_email__: dict[str, str] = {
    'Aabha Malik': 'aabhamalik30@gmail.com',
    'Manasdeep Singh': '',
    'Sarthak Srivastav': ''
}


__authors_qualifications__: dict[str, str] = {
    x: 'Btech CSE, Amity University, Noida' 
    for x in ['Aabha Malik',  'Manasdeep Singh', 'Sarthak Srivastav']    

}


__license__: str = r'''
    MIT License
    Copyright (c) 2022 Aabha Malik
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import cv2
sns.set_style('darkgrid')
sns.set(font_scale=1.2)
from PIL import Image
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder

import tensorflow as tf
import keras
from keras.models import save_model, load_model, Sequential
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam 
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 1000)
np.random.seed(0)
np.set_printoptions(suppress=True)
tf.random.set_seed(0)



#@: Data Preprocessing Step
def covid_data_preprocess() -> ImageDataGenerator:
    image_gen = ImageDataGenerator(
        rotation_range= 30,  
        width_shift_range= 0.1,
        height_shift_range= 0.1,
        rescale= 1/255,
        shear_range= 0.2,
        zoom_range= 0.2, 
        horizontal_flip= True,
        fill_mode= "nearest" 
    )
    return image_gen



#@: Model creation function
def build_model() -> keras.Model.Sequential:
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(100,100,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), input_shape=(100,100,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), input_shape=(100,100,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1,activation='sigmoid'))
    return model






#@: Driver Code
if __name__.__contains__('__main__'):
    train_path: str = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\covid_dataset\\train'
    test_path: str = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\covid_dataset\\test'
    image_gen: ImageDataGenerator = covid_data_preprocess()
    image_gen.flow_from_directory(train_path, target_size=(100, 100))
    image_gen.flow_from_directory(test_path, target_size=(100, 100))
    model = build_model()
    model.summary()
    model.compile(optimizer= 'Adam', loss= 'binary_crossentropy', metrics= ["accuracy"])
    
    batch_size: int = 1
    
    train_image_gen = image_gen.flow_from_directory(
        train_path, 
        target_size= (100,100), 
        batch_size= batch_size,
        class_mode= 'binary'
    )
    
    test_image_gen = image_gen.flow_from_directory(
        test_path, 
        target_size= (100,100), 
        batch_size= batch_size,
        class_mode= 'binary'
    )
    
    target_map: dict[str, int] = train_image_gen.class_indices
    print(target_map)
    
    #@: training the model 
    results = model.fit_generator(
        train_image_gen, 
        epochs= 100,
        verbose= 1,
        steps_per_epoch= 1,
        validation_data= test_image_gen,
        validation_steps= 1
    )
    
    #@: plotting the accuracy
    plt.plot(results.history['accuracy'])
    plt.show()
    

    
    
    
    
    
    
    
    