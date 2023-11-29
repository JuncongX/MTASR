import cv2
import os
import numpy as np
import math
import yaml

from face_detection.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from face_detection.TDDFA_ONNX import TDDFA_ONNX

config_path = 'face_detection/configs/mb1_120x120.yml'
opt = '2d_sparse'
cfgfile = yaml.load(open(config_path), Loader=yaml.SafeLoader)
face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfgfile)


def rotate_point(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def get_image_hull_mask(image_shape, image_landmarks, part="whole"):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception('get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    face_min_y = min(int_lmrks[18, 1], int_lmrks[19, 1],
                     int_lmrks[20, 1], int_lmrks[23, 1],
                     int_lmrks[24, 1], int_lmrks[25, 1])

    forehead_high = int((max(int_lmrks[7, 1], int_lmrks[8, 1], int_lmrks[9, 1]) - face_min_y) / 5)
    forehead_high_y = face_min_y - forehead_high if forehead_high < face_min_y else 0

    if part == "whole":
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(np.concatenate(
            ([[int_lmrks[26][0], forehead_high_y], [int_lmrks[17][0], forehead_high_y]], int_lmrks[0:16]))), (1,))
    elif part == "up":
        cv2.fillConvexPoly(hull_mask,
                           np.array([[int_lmrks[26][0], forehead_high_y], [int_lmrks[17][0], forehead_high_y],
                                     int_lmrks[17], int_lmrks[26]]), (1,))
    elif part == "down":
        cv2.fillConvexPoly(hull_mask,
                           np.array([[int_lmrks[12][0], int_lmrks[28][1]], [int_lmrks[4][0], int_lmrks[28][1]],
                                     [int_lmrks[4][0], int_lmrks[33][1]], [int_lmrks[12][0], int_lmrks[33][1]]]), (1,))
    return hull_mask


def merge_add_mask(img_1, mask):
    if mask is not None:
        height = mask.shape[0]
        width = mask.shape[1]
        channel_num = mask.shape[2]
        for row in range(height):
            for col in range(width):
                for c in range(channel_num):
                    if mask[row, col, c] == 0:
                        mask[row, col, c] = 0
                    else:
                        mask[row, col, c] = 255
        mask = np.array(mask, dtype="uint8")
        res_img = cv2.bitwise_and(img_1, img_1, mask=mask)
    else:
        res_img = img_1
    return res_img


def get_face_part(image, image_landmarks, part="whole"):
    if image_landmarks.shape[0] != 68:
        raise Exception('get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int16)
    face_min_y = min(int_lmrks[18, 1], int_lmrks[19, 1],
                     int_lmrks[20, 1], int_lmrks[23, 1],
                     int_lmrks[24, 1], int_lmrks[25, 1])

    forehead_high = int((max(int_lmrks[7, 1], int_lmrks[8, 1], int_lmrks[9, 1]) - face_min_y) / 5)  # 额头长度
    forehead_high_y = face_min_y - forehead_high if forehead_high < face_min_y else 0  # 额头最高y
    face_max_y = max(int_lmrks[6, 1], int_lmrks[7, 1],
                     int_lmrks[8, 1], int_lmrks[9, 1],
                     int_lmrks[10, 1])
    face_min_x = min(int_lmrks[0, 0], int_lmrks[1, 0],
                     int_lmrks[2, 0])
    face_max_x = max(int_lmrks[14, 0], int_lmrks[15, 0],
                     int_lmrks[16, 0])
    return image[forehead_high_y:face_max_y, face_min_x:face_max_x]


def cut_face(img, part="whole"):
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected')
        pass
    face = boxes[0]
    param_lst, roi_box_lst = tddfa(img, [face])
    # 人脸特征点
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    ver_lst = ver_lst[0]
    ver_lst = ver_lst.transpose(1, 0)[:, [0, 1]]

    # 眼睛两侧
    left_eye_point_x = int(ver_lst[36, 0])
    right_eye_point_x = int(ver_lst[45, 0])
    left_eye_point_y = int(ver_lst[36, 1])
    right_eye_point_y = int(ver_lst[45, 1])

    dx = right_eye_point_x - left_eye_point_x
    dy = right_eye_point_y - left_eye_point_y
    angle = math.atan2(dy, dx) * 180. / math.pi
    eye_center = ((left_eye_point_x + right_eye_point_x) // 2,
                  (left_eye_point_y + right_eye_point_y) // 2)
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotate_matrix, (img.shape[1], img.shape[0]))

    row = img.shape[0]
    for i, point in enumerate(ver_lst):
        ver_lst[i] = rotate_point(eye_center, (ver_lst[i, 0], ver_lst[i, 1]), angle, row)

    # mask = get_image_hull_mask(rotated_img.shape, ver_lst, part=part)
    # res_img = merge_add_mask(rotated_img, mask)
    res_img = get_face_part(rotated_img, ver_lst, part=part)
    return res_img


if __name__ == '__main__':
    file_path = r"TRY/face2.jfif"
    img = cv2.imread(file_path)
    res_img = cut_face(img)
    cv2.imshow("face", res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
