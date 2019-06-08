from PIL import Image
import cv2
import numpy as np
import os.path as osp
from utils import FaceNet

def get_cossim(feature1, feature2):
    return np.dot(feature1.T, feature2)/(np.linalg.norm(feature1)*np.linalg.norm(feature2))

if __name__ == '__main__':

    model_dir = './model'
    facenet = FaceNet(model_dir, gpu_id=0)

    # detect and extract feature
    image = Image.open('./test_data/jack.jpg')
    bboxes, landmarks = facenet.detect_faces(image)
    face = facenet.get_aligned_face(image, landmark=landmarks[0])
    jack_feat = facenet.extract_feature(face)

    image = Image.open('./test_data/rose.jpg')
    bboxes, landmarks = facenet.detect_faces(image)
    face = facenet.get_aligned_face(image, landmark=landmarks[0])
    rose_feat = facenet.extract_feature(face)

    image = Image.open('./test_data/titanic.jpg')
    bboxes, landmarks = facenet.detect_faces(image)
    features = facenet.easy_extract_features(image, landmarks)
    
    # match
    bbox_num = features.shape[0]
    name = []
    for i in range(bbox_num):
        rose_sim = get_cossim(features[i], rose_feat[0])
        jack_sim = get_cossim(features[i], jack_feat[0])
        print(rose_sim, jack_sim)
        if rose_sim > jack_sim:
            name.append('Rose')
        else:
            name.append('Jack')
    
    # draw and show
    cv_img = np.array(image, np.uint8)[:, : , ::-1]
    for i in range(bbox_num):
        cv_img = cv2.rectangle(
            cv_img,
            (int(bboxes[i, 0]), int(bboxes[i, 1])),
            (int(bboxes[i, 2]), int(bboxes[i, 3])),
            (0, 255, 0), 2)
        cv_img = cv2.putText(
            cv_img,
            name[i],
            (int(bboxes[i, 0]), int(bboxes[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)
        for j in range(5):
            cv_img = cv2.circle(
                cv_img,
                (int(landmarks[i, j, 0]), int(landmarks[i, j, 1])),
                2, (0, 255, 0), 2)
    cv2.imshow('haha', cv_img)
    cv2.waitKey()
