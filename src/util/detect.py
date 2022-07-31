#implementation of face detection methods
#Methods to add:SCRFD, RetinaFace, opencv, yolo5face
import cv2
from retinaface import RetinaFace
# from scrfd import SCRFD
import argparse
import numpy as np
import mediapipe
from PIL import Image
import math
from .align import align
from os.path import dirname, abspath
# from .landmarks import landmarks
import src.util.align as alignment
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
#to-do: make format output of landmarks, in order to make it easy for alignment

#due to restrictions of aligning methods, 

# overall interface, available methods = ['cv2','retina','scrfd','mediapipe']
def detect(img, method = 'cv2'):
    if(method == 'cv2'):
        return det_cv2(img)
    elif(method == 'retina'):
        return det_retina(img)
    elif(method == 'scrfd'):
        return det_scrfd(img)
    elif(method== 'mediapipe'):
        return det_mediapipe(img)
    else:
        raise(Exception('wrong detection method specified'))
        


# detection using opencv, note that opencv can only detect eye coordinator and the egde of the face, thus general normalize of the face is not possible
# hense, the only align method for opencv is rotation, though other methods are applicable, but it does not make sense(you have to detect the face using other methods)
def det_cv2(img):
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
        #convert to RGB from BGR
        cropped = cropped[:,:,::-1]
        ret_faces.append(cropped)
        
        # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #then make alignment for each face, since the posture of the faces may be different
    # if(align):
    #     for i in range(len(ret_faces)):
    #         img = ret_faces[i]
    #         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         eyes = eye_detector.detectMultiScale(gray_img)
    #         # when eyes not detected
    #         if(not len(eyes) == 2 ):
    #             continue
    #         index = 0
    #         for (eye_x, eye_y, eye_w, eye_h) in eyes:
    #             print(index)
    #             # print (eye_x, eye_y, eye_w, eye_h)
    #             if index == 0:
    #                 eye_1 = (eye_x, eye_y, eye_w, eye_h)
    #             elif index == 1:
    #                 eye_2 = (eye_x, eye_y, eye_w, eye_h)
    #             index = index + 1
    #         if(eye_1[0] < eye_2[0]):
    #             left_eye = eye_1
    #             right_eye = eye_2
    #         else:
    #             left_eye = eye_2
    #             right_eye = eye_1
    #         #caluate the center of the eyes
    #         left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    #         left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
    #         right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    #         right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
    #         landmark = {}
    #         landmark["left_eye"] = [left_eye_x,left_eye_y]
    #         landmark["right_eye"] = [right_eye_x,right_eye_y]

            

    #         ret_faces[i] = alignment.align_rotate(img,landmarks=landmark)
    
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
#ver2, rewrite with insightface
def det_scrfd(img,model='./src/scrfd_weights/scrfd_500m_kps.onnx'):
    # mynet = SCRFD(model, confThreshold, nmsThreshold)
    # srcimg = cv2.imread(img)
    # faces = mynet.detect(srcimg,align)
    # print(faces[0])
    # cv2.imshow('img',faces[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return faces
    # with default backbone, insightface would use scrfd as face detecting method
    app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
    app.prepare(ctx_id=0, det_size=(640, 640))  
    scrimg = cv2.imread(img)
    ret_faces = []
    faces = app.get(scrimg)
    for face in faces:
        bbox = face['bbox']
        y1,x1,y2,x2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cropped_face = scrimg[x1:x2,y1:y2]
        cropped_face = cropped_face[:,:,::-1]
        ret_faces.append(cropped_face)
    return ret_faces
#somehow I found this method not doing well when multiple faces in the pictures, when using world largest selfie as test image.
def det_mediapipe(img,min_detection_confidence=0.5):
    img = cv2.imread(img)
    mp_face_detection = mediapipe.solutions.face_detection
    face_detector =  mp_face_detection.FaceDetection(min_detection_confidence)
    results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # print(results.detections)
    ret_faces = []
    if results.detections:
        for face in results.detections:
            confidence = face.score
            bounding_box = face.location_data.relative_bounding_box
            y = int(bounding_box.xmin * img.shape[1])
            h = int(bounding_box.width * img.shape[1])
            x = int(bounding_box.ymin * img.shape[0])
            w = int(bounding_box.height * img.shape[0])
            cropped_face = img[x:x+w,y:y+h]
            cropped_face = cropped_face[:,:,::-1]
            ret_faces.append(cropped_face)
    return ret_faces


# faces = det_mediapipe('..//selfie.jpg')
# print(len(faces))
# img = cv2.imread('selfie.jpg')
# print(img)
# cv2.imshow('img',faces[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

det_scrfd('test.jpg')