import numpy as np
import cv2 as cv
import os
import zipfile
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

data_dir = '/ACRIMA'
# Load the ACRIMA dataset
def load_images_and_labels(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for image_filename in os.listdir(label_path):
            img_path = os.path.join(label_path, image_filename)
            try:
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                images.append(img_array)
                if label.lower() == 'glaucoma':
                    labels.append(1)  # 1 for glaucoma
                else:
                    labels.append(0)  # 0 for normal
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
    return np.array(images), np.array(labels)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE_DIR, 'ACRIMA', 'PARTITIONED')
train_dir = os.path.join(DIR, 'Training')
test_dir = os.path.join(DIR, 'Testing')


X_train_cnn2, y_train_cnn2 = load_images_and_labels(train_dir)
X_test_cnn2, y_test_cnn2 = load_images_and_labels(test_dir)

X_train_cnn2 = X_train_cnn2 / 255.0  # Normalize and add channel dimension
X_test_cnn2 = X_test_cnn2 / 255.0

# One-hot encode labels for CNN training
y_train_cat2 = to_categorical(y_train_cnn2)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the CNN model
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

# Train the CNN model
hist2= cnn2.fit(X_train_cnn2, y_train_cat2, epochs=10, validation_split=0.2)


cnn2.evaluate(X_test_cnn2, to_categorical(y_test_cnn2))
print("Accuracy:" + str(cnn2.evaluate(X_test_cnn2, to_categorical(y_test_cnn2))[1]))
cnn2.save('cnn_acrima.h5')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict the classes of the test set
y_pred_cnn2 = np.argmax(cnn2.predict(X_test_cnn2), axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_cnn2, y_pred_cnn2)
print(f'Accuracy: {accuracy:.4f}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_cnn2, y_pred_cnn2)
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Glaucoma'], yticklabels=['Normal', 'Glaucoma'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
class_report = classification_report(y_test_cnn2, y_pred_cnn2, target_names=['Normal', 'Glaucoma'])
print('Classification Report:')
print(class_report)

import pandas as pd
results=[]
true_positive=conf_matrix[0][0]
false_negative=conf_matrix[0][1]
false_positive=conf_matrix[1][0]
true_negative=conf_matrix[1][1]
accuracy = accuracy_score(y_test_cnn2, y_pred_cnn2)
precision=true_positive/(true_positive+false_positive)
recall=true_positive/(true_positive+false_negative)
f1_score=2*(precision*recall)/(precision+recall)
results.append({
        "Model name":"CNN",
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1 score":f1_score
})
data=pd.DataFrame(results)

from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_glaucoma(image_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        predicted_class = np.argmax(cnn2.predict(preprocessed_image), axis=1)[0]
        if predicted_class == 1:
            return "Glaucoma"
        else:
            return "Normal"
    else:
        return "Error in processing image"

        
def predict_and_plot(path):
    prediction = predict_glaucoma(path)
    return prediction