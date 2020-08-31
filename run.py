import cv2 
import tensorflow as tf
import numpy as np

tf.config.experimental.set_visible_devices([], 'GPU')

#labels
labels={0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

#Load the model from h5 file 
model=tf.keras.models.load_model('emotion.h5')

#Preprocess the image
def preprocess(img):
	img=cv2.resize(img,(64,64))
	img.astype('float32')
	img=img/255
	img=np.expand_dims(img,-1)
	img=np.expand_dims(img,0)
	
	return labels[np.argmax(model.predict(img))]
	
#Load the cascade model into cv2	
cascade_path='cascade_haar.xml'
faceNet=cv2.CascadeClassifier(cascade_path)

#Video Camera object  1 for external webcam 0 for internal
cam=cv2.VideoCapture(1) 

#Output image 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    ret,img=cam.read()

    #Convert to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Detect faces using HaarCascade
    faces=faceNet.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(64,64))

    #For each face process predictions
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

        #Preprocess and predict
        part=gray[y:y+h,x:x+w]
        prediction=preprocess(part)

        #Set out confidence score
        cv2.putText(img,str("Predicted: "+prediction),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)

    out.write(img)
    cv2.imshow('Video',img)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()