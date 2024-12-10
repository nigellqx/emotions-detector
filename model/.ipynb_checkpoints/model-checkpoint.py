import numpy as np
import cv2
import joblib
import json
import matplotlib
import os
import shutil
import pywt
import pandas as pd
from better_bing_image_downloader import downloader
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#Images from kaggle

face_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/haarcascade_eye.xml')

def get_cropped_face_if_valid(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 5)
    if len(faces) == 1:
        cropped_face = img_gray[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
        return cropped_face

path_to_images = './model/Images/'
path_to_cropped_images = './model/Images/cropped/'

img_paths = []
for type in os.scandir(path_to_images):
    if type.is_dir():
        img_paths.append(type.path)

if os.path.exists(path_to_cropped_images):
    shutil.rmtree(path_to_cropped_images)
os.mkdir(path_to_cropped_images)

cropped_img_paths = []
emotions_dict = {}

for path in img_paths:
    emotions_name = path.split('/')[-1]
    emotions_dict.update({emotions_name: []})
    count = 1
    for img in os.scandir(path):
        if count == 500:
            break
        cropped = get_cropped_face_if_valid(img.path)
        if cropped is not None:
            cropped_folder = path_to_cropped_images + emotions_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_img_paths.append(cropped_folder)
            
            cropped_image_name = emotions_name + str(count) + ".jpg"
            cropped_image_path = cropped_folder + "/" + cropped_image_name
            cv2.imwrite(cropped_image_path, cropped)
            emotions_dict[emotions_name].append(cropped_image_path)
            count += 1

#From stackoverflow
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #Convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    #Convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    #Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    #Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

emotions_numbering = {}
count = 0
for emotion in emotions_dict.keys():
    emotions_numbering[emotion] = count
    count += 1

x=[] 
y=[]

for emotion, image_path in emotions_dict.items():
    for image in image_path:
        print(image_path)
        img = cv2.imread(image)
        if img is None:
            continue
        img_scale = cv2.resize(img, (32,32))
        img_w2d = w2d(img, 'db1',5)
        img_w2d_scale = cv2.resize(img_w2d, (32,32))
        combined_img = np.vstack((img_scale.reshape(32*32*3,1), img_w2d_scale.reshape(32*32,1)))
        x.append(combined_img)
        y.append(emotions_numbering[emotion])
X = np.array(x).reshape(len(x),4096).astype(float)
print("test1")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scalar', StandardScaler()), ('svc', SVC(kernel = 'rbf',C=10))])
pipe.fit(X_train, y_train)

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params': {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}

scores =[]
best_estimators = {}
for algo, mp in model_params.items():
    print(algo)
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train,y_train)
    scores.append({
        'model':algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params'])
print(df[['model']])
print(df[['best_score']])
print(df[['best_params']])

print(best_estimators['svm'].score(X_test,y_test))
print(best_estimators['random_forest'].score(X_test,y_test))
print(best_estimators['logistic_regression'].score(X_test,y_test))
best_clf = best_estimators['svm']
cm = confusion_matrix(y_test, best_clf.predict(X_test))
print(cm)
best_clf = best_estimators['svm']

#Save the model
joblib.dump(best_clf, 'saved_model.pkl')

#Save emotion dictionary
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(emotions_numbering))