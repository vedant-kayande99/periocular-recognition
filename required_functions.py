# demonstrate face detection on 5 Celebrity Faces Dataset
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from numpy import asarray
from keras_vggface.utils import preprocess_input
import cv2

def get_embedding(face, model):
    img_data = face.astype('float32')
    img_data = expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)    
    return vgg16_feature

eye_cascadeR = cv2.CascadeClassifier('cascades/ojoD.xml')
eye_cascadeL = cv2.CascadeClassifier('cascades/ojoI.xml')

def get_eye(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	eye_cords = eye_cascadeL.detectMultiScale(gray, 1.01)
	for (ex, ey, ew, eh) in eye_cords:
		x1, y1 = abs(ex), abs(ey)
		x2, y2 = ex+ew, ey+eh

	if list(eye_cords) == []:
		eye_cords = eye_cascadeR.detectMultiScale(gray, 1.01)
		for (ex, ey, ew, eh) in eye_cords:
			x1, y1 = abs(ex), abs(ey)
			x2, y2 = ex+ew, ey+eh

	eye = img[y1:y2, x1:x2]
	return eye