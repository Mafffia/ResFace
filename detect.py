#implementation of face detection methods
#Methods to add:SCRFD, RetinaFace, opencv, yolo5face
import cv2
from retinaface import RetinaFace
from scrfd import SCRFD
import argparse
import numpy as np
import mediapipe

# detection using opencv, note that opencv can only detect eye coordinator and the egde of the face, thus general normalizatio of the face is not possible
def det_cv2(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")
    img = cv2.imread(img)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(faces)
    # Draw rectangle around the faces
    ret_faces = []
    for (x, y, w, h) in faces:
        cropped = img[y:y+h,x:x+w]
        ret_faces.append(cropped)
        
        # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    return ret_faces
   

# detection using RetinaFace
def det_retina(img):
    faces = RetinaFace.extract_faces(img_path = img, align = False)
    # for face in faces:
        # plt.imshow(face)
        # plt.show()
        # print(face.shape)
    return faces

#detect using scrfd, implemented in opencv, thanks to https://github.com/hpc203/scrfd-opencv
def det_scrfd(img,model='./src/scrfd_weights/scrfd_500m_kps.onnx',confThreshold=0.5,nmsThreshold=0.5):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--imgpath', type=str, default='s_l.jpg', help='image path')
    # parser.add_argument('--onnxmodel', default='weights/scrfd_500m_kps.onnx', type=str, choices=['weights/scrfd_500m_kps.onnx', 'weights/scrfd_2.5g_kps.onnx', 'weights/scrfd_10g_kps.onnx'], help='onnx model')
    # parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    # parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    # args = parser.parse_args()

    mynet = SCRFD(model, confThreshold, nmsThreshold)
    srcimg = cv2.imread(img)
    faces = mynet.detect(srcimg)
    return faces

def det_mediapipe(img,min_detection_confidence=0.6):
    img = cv2.imread(img)
    mp_face_detection = mediapipe.solutions.face_detection
    face_detector =  mp_face_detection.FaceDetection(min_detection_confidence)
    results = face_detector.process(img)
    print(results.detections)
    ret_faces = []
    if results.detections:
        for face in results.detections:
            confidence = face.score
            bounding_box = face.location_data.relative_bounding_box
            
            x = int(bounding_box.xmin * img.shape[1])
            w = int(bounding_box.width * img.shape[1])
            y = int(bounding_box.ymin * img.shape[0])
            h = int(bounding_box.height * img.shape[0])
            
            ret_faces.append(img[y:y+h,x:x+w])

            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness = 2)
            # landmarks = face.location_data.relative_keypoints
 
            # right_eye = (int(landmarks[0].x * img.shape[1]), int(landmarks[0].y * img.shape[0]))
            # left_eye = (int(landmarks[1].x * img.shape[1]), int(landmarks[1].y * img.shape[0]))
            # nose = (int(landmarks[2].x * img.shape[1]), int(landmarks[2].y * img.shape[0]))
            # mouth = (int(landmarks[3].x * img.shape[1]), int(landmarks[3].y * img.shape[0]))
            # right_ear = (int(landmarks[4].x * img.shape[1]), int(landmarks[4].y * img.shape[0]))
            # left_ear = (int(landmarks[5].x * img.shape[1]), int(landmarks[5].y * img.shape[0]))
            
            # cv2.circle(img, right_eye, 15, (0, 0, 255), -1)
            # cv2.circle(img, left_eye, 15, (0, 0, 255), -1)
            # cv2.circle(img, nose, 15, (0, 0, 255), -1)
            # cv2.circle(img, mouth, 15, (0, 0, 255), -1)
            # cv2.circle(img, right_ear, 15, (0, 0, 255), -1)
            # cv2.circle(img, left_ear, 15, (0, 0, 255), -1)
    cv2.imshow('img',ret_faces[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
det_cv2('/home/mafffia/Desktop/ResFace/test.jpg')