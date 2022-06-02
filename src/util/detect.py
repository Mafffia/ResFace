#implementation of face detection methods
#Methods to add:SCRFD, RetinaFace, opencv, yolo5face
import cv2
from retinaface import RetinaFace
from util.scrfd import SCRFD
import argparse
import numpy as np
import mediapipe
from PIL import Image
import math
from util.align import align as alignment


#to-do: make format output of landmarks, in order to make it easy for alignment



# detection using opencv, note that opencv can only detect eye coordinator and the egde of the face, thus general normalize of the face is not possible
# hense, the only align method for opencv is rotation, though other methods are applicable, but it does not make sense(you have to detect the face using other methods)
def det_cv2(img,align = True):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    rimg = cv2.imread(img)
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
    #then make alignment for each face, since the posture of the faces may be different
    if(align):
        for i in range(len(ret_faces)):
            img = ret_faces[i]
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eyes = eye_detector.detectMultiScale(gray_img)
 
            index = 0
            for (eye_x, eye_y, eye_w, eye_h) in eyes:
                if index == 0:
                    eye_1 = (eye_x, eye_y, eye_w, eye_h)
                elif index == 1:
                    eye_2 = (eye_x, eye_y, eye_w, eye_h)
                index = index + 1
            if(eye_1[0] < eye_2[0]):
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1
            #caluate the center of the eyes
            left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
            left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
            right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
            right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
            landmark = {}
            landmark["left_eye"] = [left_eye_x,left_eye_y]
            landmark["right_eye"] = [right_eye_x,right_eye_y]
            # if left_eye_y > right_eye_y:
            #     point_3rd = (right_eye_x, left_eye_y)
            #     direction = -1 #rotate same direction to clock
            #     # print("rotate to clock direction")
            # else:
            #     point_3rd = (left_eye_x, right_eye_y)
            #     direction = 1 #rotate inverse direction of clock
            #     # print("rotate to inverse clock direction")
            # # cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)
 
            # # cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)
            # # cv2.line(img,left_eye_center, point_3rd,(67,67,67),2)
            # # cv2.line(img,right_eye_center, point_3rd,(67,67,67),2)
            
            # a = euclidean_distance(left_eye_center, point_3rd)
            # b = euclidean_distance(right_eye_center, left_eye_center)
            # c = euclidean_distance(right_eye_center, point_3rd)
            # cos_a = (b*b + c*c - a*a)/(2*b*c)
            # angle = np.arccos(cos_a)
            # angle = (angle * 180) / math.pi
            # if direction == -1:
            #     angle = 90 - angle
            # new_img = Image.fromarray(img)
            # new_img = np.array(new_img.rotate(direction * angle))
            ret_faces[i] = alignment(img,method='rotate',landmarks=landmark)
    cv2.imshow('img',ret_faces[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
def det_scrfd(img,model='./src/scrfd_weights/scrfd_500m_kps.onnx',confThreshold=0.5,nmsThreshold=0.5,align=True):
    mynet = SCRFD(model, confThreshold, nmsThreshold)
    srcimg = cv2.imread(img)
    faces = mynet.detect(srcimg,align)
    cv2.imshow('img',faces[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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