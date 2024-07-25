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
DIR = os.path.join(BASE_DIR, 'ACRIMA','PARTITIONED','Testing')
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#Added cnn model later on
cnn2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_train_cnn2)), activation='softmax')
])
cnn2.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

models = {
    'svm': {
        'model': SVC() 
    },
    'random_forest': {
        'model': RandomForestClassifier()
    },
    'logistic_regression' : {
        'model': LogisticRegression(max_iter=1000)
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB()
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB()
    },
    'decision_tree': {
        'model': DecisionTreeClassifier()
    },
    'Adaboost with DecisionTree': {
        'model': AdaBoostClassifier(estimator=DecisionTreeClassifier(),n_estimators=10)
    },
    'Adaboost with Logistic Regression': {
        'model': AdaBoostClassifier(estimator=LogisticRegression(max_iter=1000),n_estimators=10)
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=10)
    },
    'CNN': {
        'model': cnn2
    }

}


results=[]
for model_name,classifier in models.items():
    model=classifier['model']

    if model_name=='CNN':
        model.fit(scaled_x,y_test,epochs=10,batch_size=10)
    else:
        model.fit(scaled_x,y)

    if model_name=='CNN':
        accuracy = accuracy_score(y_test,np.argmax(model.predict(scaled_x_test), axis=1))
    else:
        accuracy = model.score(scaled_x_test,y_test)

    if model_name=='CNN':
        y_predicted=np.argmax(model.predict(scaled_x_test), axis=1)
    else:
        y_predicted=model.predict(scaled_x_test)
        
    cm=confusion_matrix(y_test,y_predicted)
    true_positive=cm[0][0]
    false_negative=cm[0][1]
    false_positive=cm[1][0]
    true_negative=cm[1][1]
    precision=true_positive/(true_positive+false_positive)
    recall=true_positive/(true_positive+false_negative)
    f1_score=2*(precision*recall)/(precision+recall)
    results.append({
        'Model name':model_name,
        'Accuracy':accuracy,
        'Precision':precision,
        'Recall':recall,
        'F1 score':f1_score
    })
data=pd.DataFrame(results)


def getdata():
    st.write(data)
    return data
