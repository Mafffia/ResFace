from retinaface import RetinaFace
from insightface.app import FaceAnalysis
import cv2
import mediapipe
"""
In this project, the landmark array we use is as following:
landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]]

due to different detect method have different numbers and format of landmarks, thus I rewrote this file in order it could support different detection methods.
"""

"""
faces = RetinaFace.detect_faces(img_path = 'test.jpg')
faces = {'face_1': {'score': 0.9989180564880371,
  'facial_area': [86, 76, 167, 192],
  'landmarks': {'right_eye': [109.999374, 118.28145],
   'left_eye': [146.62201, 117.99363],
   'nose': [129.83868, 135.41493],
   'mouth_right': [111.90126, 157.98032],
   'mouth_left': [146.38423, 157.41806]}}}

"""
def retina_landmarks(img):
    # notice that load in file
    faces = RetinaFace.detect_faces(img_path=img)
    landmarks = []
    for facekey in faces.keys():
        lmks = faces[facekey]['landmarks']
        right_eye = lmks['left_eye']
        left_eye = lmks['right_eye']
        nose = lmks['nose']
        #notice that the keypoints in retinaface is oppsite to the view
        mouth_right = lmks['mouth_left']
        mouth_left = lmks['mouth_right']
        landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]]

        landmarks.append(landmark)
    
    return landmarks




"""
we are using scrfd from insightface library, in order to save time :)

from insightface.app import FaceAnalysis

app = FaceAnalysis(allowed_modules=['detection']) 
app.prepare(ctx_id=0, det_size=(640, 640))  
scrimg = cv2.imread('test.jpg')
faces = app.get(scrimg)
faces = [{'bbox': array([ 85.47469,  76.81839, 165.40952, 190.46407], dtype=float32),
  'kps': array([[109.66368 , 118.162125],
         [146.80638 , 116.46279 ],
         [132.13588 , 137.413   ],
         [113.007034, 157.01833 ],
         [147.43106 , 155.405   ]], dtype=float32),
  'det_score': 0.8473418}]



"""
def scrfd_landmarks(img,model='./src/scrfd_weights/scrfd_500m_kps.onnx'):
    app = FaceAnalysis(allowed_modules=['detection']) 
    app.prepare(ctx_id=0, det_size=(640, 640))  
    scrimg = img
    if(type(scrimg) == str):
        scrimg = cv2.imread(img)
    landmarks = []
    faces = app.get(scrimg)

    for face in faces:
        lmks = face['kps']
        left_eyes = lmks[0]
        right_eyes = lmks[1]
        nose = lmks[2]
        mouth_left = lmks[3]
        mouth_right = lmks[4]
        landmark = [left_eyes,
                   right_eyes,
                   nose,
                   mouth_left,
                   mouth_right]
        landmarks.append(landmark)
    return landmarks


def mediapipe_landmarks(img):
    if(type(img) == str):
        img = cv2.imread(img)
    mp_face_detection = mediapipe.solutions.face_detection
    face_detector =  mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # print(results.detections)
    landmarks = []
    for face in results.detections:
        kps = []
        for kp in face.location_data.relative_keypoints:
            kps.append([(kp.x*img.shape[1]),(kp.y*img.shape[0])])
        a = kps[0]
        kps[0] = kps[1]
        kps[1] = a
        a = kps[3]
        kps[3] = kps[4]
        kps[4] = a
        landmarks.append(kps)
    return landmarks