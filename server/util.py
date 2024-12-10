import joblib
import json
import numpy as np
import base64
import cv2
import pywt

__emotion_to_index = {}
__index_to_emotion = {}

__model = None

def classify_emotion(image_base64_data, file_path=None):

    images = get_cropped_face_if_valid(file_path, image_base64_data)

    result = []
    for image in images:
        image_scale = cv2.resize(image, (32, 32))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coefficients = pywt.dwt2(image_gray, 'db1')
        LL, (LH, HL, HH) = coefficients
        image_har = LL
        
        image_har_scale = cv2.resize(image_har, (32, 32))
        
        combined_img = np.vstack((image_scale.reshape(32 * 32 * 3, 1), image_har_scale.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        
        result.append({
            'class': index_to_emotion(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __emotion_to_index
        })

    return result

def index_to_emotion(index):
    return __index_to_emotion[index]

def load_saved_artifacts():
    global __emotion_to_index
    global __index_to_emotion

    with open("./artifacts/class_dictionary.json", "r") as f:
        __emotion_to_index = json.load(f)
        __index_to_emotion = {v:k for k,v in __emotion_to_index.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_face_if_valid(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 2)
    cropped_faces = []
    if len(faces) != 0:
        (x,y,w,h) = faces[0]
        cropped_face = img[y:y+h, x:x+w]
        cropped_faces.append(cropped_face)
    return cropped_faces

if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_emotion(None, "./test.jpg"))