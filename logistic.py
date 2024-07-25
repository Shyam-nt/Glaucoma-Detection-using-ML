#importing required modules

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import cv2 as cv
import sklearn


#accessing folders inside the specified path
folders=[]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE_DIR, 'ACRIMA', 'PARTITIONED','Training')
for folder in os.listdir(DIR):
    folders.append(folder)

classes=['No Glaucoma','Glaucoma']

labels=[]
features=[]

#creating x(features) and y(labels) data
for folder in folders:
    path=os.path.join(DIR,folder)
    for image in os.listdir(path):
        if folder=='normal':
            labels.append(0)
        else:
            labels.append(1)
        image_path=os.path.join(path,image)
        img=cv.imread(image_path)
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        resized=cv.resize(gray,(290,290))
        features.append(resized)

#converting "features" into numpy array
x=np.array(features)

#converting "labels" into numpy array
y=np.array(labels)

#creating x_test and y_test data
test_features=[]
test_labels=[]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE_DIR, 'ACRIMA', 'PARTITIONED','Testing')
for folder in folders:
    path=os.path.join(DIR,folder)
    for image in os.listdir(path):
        if folder=='normal':
            test_labels.append(0)
        else:
            test_labels.append(1)
        image_path=os.path.join(path,image)
        img=cv.imread(image_path)
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        resized=cv.resize(gray,(290,290))
        test_features.append(resized)

x_test=np.array(test_features)

y_test=np.array(test_labels)


#reshaping images of 290*290 pixels into 84100
#scaling values by dividing with 255 in order to scale them between 0 and 1
scaled_x=x.reshape(-1,290*290)/255

scaled_x_test=x_test.reshape(-1,290*290)/255


#importing Logistic regression from scikit-learn
from sklearn.linear_model import LogisticRegression
#creating an object for LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(scaled_x,y)
#test data
model.score(scaled_x_test,y_test)
#predicting for all the test images
y_predicted=model.predict(scaled_x_test)


def predict_and_plot(path):
    img=cv.imread(path)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    resized=cv.resize(gray,(290,290))
    x_var=np.array(resized)
    x_var=x_var.reshape(-1,290*290)/255
    predicted=model.predict(x_var)
    plt.figure(figsize=(2.78, 2.78))
    plt.imshow(img)
    plt.title(classes[predicted[0]])
    st.pyplot(plt.gcf())

    
    