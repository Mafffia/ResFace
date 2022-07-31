"""This is the py file for face align, we try to use the align tech in ArcFace, rather than the align in DeepFace"""
from insightface.utils import face_align
import cv2
from insightface.app import FaceAnalysis
import mtcnn
import numpy as np
from skimage import transform as trans
import cv2

def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm

def detect_landmark(image, detector):
    '''
    image as numpy format with RGB format
    note that cv2 read is BGR format
    '''
    face = detector.detect_faces(image)[0]

    #draw points
    left_eye = face["keypoints"]["left_eye"]
    right_eye = face["keypoints"]["right_eye"]
    nose = face["keypoints"]["nose"]
    mouth_left = face["keypoints"]["mouth_left"]
    mouth_right = face["keypoints"]["mouth_right"]
    landmark = [[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]]
    return landmark



src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.AffineTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M,_ = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


""""Input: the dir of the file
    Output: cv2 image with already aligned face
"""
def get_face_align(img_file):
    detector = mtcnn.MTCNN()
    if(type(img_file) == str):
        img_file = cv2.imread(img_file)
    img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)  # To RGB
    landmark = detect_landmark(img, detector)
    wrap = norm_crop(img, np.array(landmark), image_size=112, mode='arcface')
    return wrap