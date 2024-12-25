import cv2
import numpy as np
from tensorflow.keras.models import load_model
from google.colab.patches import cv2_imshow

model = load_model('model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread("test.jpg")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grey, 1.1, minNeighbors=5)
emotions_label = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
#emotions_label = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

for (x, y, w, h) in faces:
    face_image = grey[y:y+h, x:x+w]
    resized_face = cv2.resize(face_image, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

    prediction = model.predict(reshaped_face)
    label_index = np.argmax(prediction, axis=1)[0]
    emotion = emotions_label[label_index]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

cv2.imwrite("output.jpg", img)
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
