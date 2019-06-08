import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage import transform as trans
from .get_nets import PNet, RNet, ONet, resnet50
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess
import math
import PIL
from PIL import Image
import os.path as osp
import time
import cv2


class FaceNet(object):
    def __init__(self, model_dir, gpu_id=-1):
        self.pnet = PNet(osp.join(model_dir, 'pnet.npy'))
        self.rnet = RNet(osp.join(model_dir, 'rnet.npy'))
        self.onet = ONet(osp.join(model_dir, 'onet.npy'))
        self.verifynet = resnet50()
        self.verifynet.load_state_dict(
            torch.load(osp.join(model_dir, 'verify-res50.pth.tar')))
        self.gpu_id = gpu_id
        if self.gpu_id >= 0:
            self.pnet.cuda(self.gpu_id)
            self.rnet.cuda(self.gpu_id)
            self.onet.cuda(self.gpu_id)
            self.verifynet.cuda(self.gpu_id)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.verifynet.eval()

        # normalizer for verification
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25])
        self.transform_verify = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            normalize,
        ])

        # mean pose for alignment
        self._mean_pose = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)


    def detect_faces(
        self, image,
        min_face_size=None,
        thresholds=[0.6, 0.7, 0.8],
        nms_thresholds=[0.7, 0.7, 0.6]):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number or None (None means dynamic min size).
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # tic = time.time()

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)
        if not min_face_size:
            min_face_size = min_length*0.05

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size/min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = self.run_first_stage(image, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        if len(bounding_boxes) == 0:
            return [], []
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        if len(img_boxes) == 0:
            return [], []
        if self.gpu_id >= 0:
            img_boxes = torch.from_numpy(img_boxes).cuda(self.gpu_id)
            output = self.rnet(img_boxes)
            offsets = output[0].detach().cpu().numpy()  # shape [n_boxes, 4]
            probs = output[1].detach().cpu().numpy()  # shape [n_boxes, 2]
        else:
            img_boxes = torch.from_numpy(img_boxes)
            output = self.rnet(img_boxes)
            offsets = output[0].data.numpy()  # shape [n_boxes, 4]
            probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])

        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3
        tic = time.time()
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)

        if len(img_boxes) == 0:
            return [], []
        if self.gpu_id >= 0:
            img_boxes = torch.from_numpy(img_boxes).cuda(self.gpu_id)
            output = self.onet(img_boxes)
            landmarks = output[0].detach().cpu().numpy()  # shape [n_boxes, 10]
            offsets = output[1].detach().cpu().numpy()  # shape [n_boxes, 4]
            probs = output[2].detach().cpu().numpy()  # shape [n_boxes, 2]
        else:
            img_boxes = torch.from_numpy(img_boxes)
            output = self.onet(img_boxes)
            landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].data.numpy()  # shape [n_boxes, 4]
            probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]
        new_landmarks = np.zeros((landmarks.shape[0], 5, 2))
        new_landmarks[:, :, 0] = landmarks[:, 0:5]
        new_landmarks[:, :, 1] = landmarks[:, 5:10]

        return bounding_boxes, new_landmarks

    def run_first_stage(self, image, scale, threshold):
        """Run P-Net, generate bounding boxes, and do NMS.

        Arguments:
            image: an instance of PIL.Image.
            scale: a float number,
                scale width and height of the image by this number.
            threshold: a float number,
                threshold on the probability of a face when generating
                bounding boxes from predictions of the net.

        Returns:
            a float numpy array of shape [n_boxes, 9],
                bounding boxes with scores and offsets (4 + 1 + 4).
        """

        # scale the image and convert it to a float array
        width, height = image.size
        sw, sh = math.ceil(width*scale), math.ceil(height*scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, 'float32')

        if self.gpu_id >= 0:
            img = torch.from_numpy(_preprocess(img)).cuda(self.gpu_id)
            output = self.pnet(img)
            probs = output[1].detach().cpu().numpy()[0, 1, :, :]
            offsets = output[0].detach().cpu().numpy()
        else:
            img = torch.from_numpy(_preprocess(img))
            output = self.pnet(img)
            probs = output[1].data.numpy()[0, 1, :, :]
            offsets = output[0].data.numpy()
        boxes = self._generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]
    
    def extract_feature(self, face):
        """
        extract verification feature of one aligned face
        """
        face = self.transform_verify(face).unsqueeze(0)
        if self.gpu_id >= 0:
            feature = self.verifynet(face.cuda(self.gpu_id))
        else:
            feature = self.verifynet(face)
        return feature.detach().cpu().numpy()
    
    def extract_feature_batch(self, faces):
        """
        extract verification feature of one aligned face
        """
        input_list = []
        for face in faces:
            input_list.append(self.transform_verify(face).unsqueeze(0))
        inputs = torch.cat(input_list, dim=0)
        if self.gpu_id >= 0:
            feature = self.verifynet(inputs.cuda(self.gpu_id))
        else:
            feature = self.verifynet(inputs)
        return feature.detach().cpu().numpy()
    
    def easy_extract_features(self, image, landmarks):
        face_num = len(landmarks)
        face_list = []
        for i in range(face_num):
            landmark = landmarks[i]
            face = self.get_aligned_face(image, landmark)
            face_list.append(face)
        feature = self.extract_feature_batch(face_list)
        return feature

    def _generate_bboxes(self, probs, offsets, scale, threshold):
        """Generate bounding boxes at places
        where there is probably a face.

        Arguments:
            probs: a float numpy array of shape [n, m].
            offsets: a float numpy array of shape [1, 4, n, m].
            scale: a float number,
                width and height of the image were scaled by this number.
            threshold: a float number.

        Returns:
            a float numpy array of shape [n_boxes, 9]
        """

        # applying P-Net is equivalent, in some sense, to
        # moving 12x12 window with stride 2
        stride = 2
        cell_size = 12

        # indices of boxes where there is probably a face
        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        # transformations of bounding boxes
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        # they are defined as:
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # x1_true = x1 + tx1*w
        # x2_true = x2 + tx2*w
        # y1_true = y1 + ty1*h
        # y2_true = y2 + ty2*h

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images
        # so we need to rescale bounding boxes back
        bounding_boxes = np.vstack([
            np.round((stride*inds[1] + 1.0)/scale),
            np.round((stride*inds[0] + 1.0)/scale),
            np.round((stride*inds[1] + 1.0 + cell_size)/scale),
            np.round((stride*inds[0] + 1.0 + cell_size)/scale),
            score, offsets
        ])
        # why one is added?

        return bounding_boxes.T

    def get_aligned_face(
        self, image, landmark=None, bbox=None,
        image_size=(112, 112), margin=44):
        if landmark is not None:
            dst = landmark.astype(np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(self._mean_pose, dst)
            # tform.estimate(dst, self._mean_pose)
            matrix = tform.params[0:2, :]
            ret = image.transform(
                image_size,
                Image.AFFINE,
                matrix.flatten(),
                resample=Image.BILINEAR)
            # warped = cv2.warpAffine(
            #     np.array(image, np.uint8) , matrix,
            #     (image_size[1],image_size[0]), borderValue=0.0)
            # ret = Image.fromarray(warped)
        elif bbox is not None:
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(bbox[0]-margin/2, 0)
            bb[1] = np.maximum(bbox[1]-margin/2, 0)
            bb[2] = np.minimum(bbox[2]+margin/2, image.size[0])
            bb[3] = np.minimum(bbox[3]+margin/2, image.size[1])
            ret = image.crop(bb)
            if len(image_size) > 0:
                ret = ret.resize(image_size, Image.BILINEAR)
        else:
            raise ValueError('No landmarks and bbox!')
        return ret