"""
align.py
In ResFace, we would perform three align methods:
1. rotate according eye coordinate
2. affline transform according stand face
3. deep neural network for alignment
"""
import math
from PIL import Image
import numpy as np


#methods = ['rotate','affline','dnn'], for first two mehods, landmarks should also be provided
def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def align_rotate(img,landmarks):
    left_eye_center = (landmarks[0][0],landmarks[0][1])
    right_eye_center = (landmarks[1][0],landmarks[1][1])
    left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
    right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        # print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        # print("rotate to inverse clock direction")
    # cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)

    # cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)
    # cv2.line(img,left_eye_center, point_3rd,(67,67,67),2)
    # cv2.line(img,right_eye_center, point_3rd,(67,67,67),2)

    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, left_eye_center)
    c = euclidean_distance(right_eye_center, point_3rd)
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = np.arccos(cos_a)
    angle = (angle * 180) / math.pi
    if direction == -1:
        angle = 90 - angle
    new_img = Image.fromarray(img)
    new_img = np.array(new_img.rotate(direction * angle))
    
    return new_img




def align(img,method = 'rotate',landmarks=None):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    #only affline method needs
    if(method == 'affline'):
        nose = landmarks["nose"]
        mouth_left = landmarks["mouth_left"]
        mouth_right = landmarks["mouth_right"]

        landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]]
    elif(method == 'rotate'):
        landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]]]
        return align_rotate(img, landmark)
    
    