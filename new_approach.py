import numpy as np
import cv2
from matplotlib import pyplot as plt
from segmentation import segmentation, mark_boundaries
import copy
from scipy.spatial import distance


image_sequence = []
# read cars2
path = "/Users/yifeima/Documents/CMPUT414/BackgroundSubtractionForMovingCamera/moseg_dataset/"
cap = cv2.VideoCapture(path + "/cars4/cars4_%02d.jpg")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image_sequence.append(frame)

# for frame in image_sequence:
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


image_sift_sequence = []

sift = cv2.xfeatures2d.SURF_create()
for image in image_sequence:
    kp, des = sift.detectAndCompute(image, None)
    # print(type(kp))
    # print("\n\n\n")
    # print(type(des))
    # break
    image_sift_sequence.append((kp, des))
    print(len(kp))
    # out_image = image
    # img = cv2.drawKeypoints(image, kp, out_image)
    # cv2.imshow("features", img)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break


# image_label_sequence = []
# for image in image_sequence:
#     labels = segmentation(image, 3)
#     image_label_sequence.append(labels)
#
# np.save('segmentation.npy', np.asarray(image_label_sequence))

image_label_sequence = np.load("segmentation.npy")
outputs = []
for i in range(len(image_sequence)):
    image = image_sequence[i]
    labels = image_label_sequence[i]
    # j = 0
    for label in labels:
        outputs.append(mark_boundaries(image, label))
        # cv2.imwrite('segs_{}_{}.jpg'.format(i, j), mark_boundaries(image, label))
        # j += 1


# i = 0
for img in outputs:
    # img = cv2.convertScaleAbs(img, alpha=(255.0))
    # cv2.imwrite('segs_{}.jpg'.format(i), img, )
    cv2.imshow("segmentation", img)
    # i += 1
    # if i >= 10: exit()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# label = image_label_sequence[0][2]
# image = image_sequence[0]
# print(np.max(label))
# mask = label == 3
# mask = np.array(mask, dtype=np.uint8)
#
# sift = cv2.xfeatures2d.SIFT_create(nfeatures=10000)
# kp, des = sift.detectAndCompute(image, mask=mask)
# out_image = image
# img = cv2.drawKeypoints(image, kp, out_image)
# print(len(kp))
# cv2.imshow("sift", img)
# cv2.waitKey(10000)


def get_trans_matrices(current_keys, current_descriptors, next_keys, next_descriptors):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(current_descriptors, next_descriptors, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([current_keys[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([next_keys[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M

    # if len(good) > 3:
    #     src_pts = np.float32([current_keys[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([next_keys[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #
    #     M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #
    #     return M
    # else:
    #     return None


def get_new_coord(x, y, matrix):
    p = (x, y)  # original point
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    p_after = (int(px), int(py))  # after transformation
    return p_after


# original_features_sequence = copy.deepcopy(image_sift_sequence)

prob_dict = {}
feature_key_sequence = []
feature_isBackground_sequence = []
for image_features in image_sift_sequence:
    temp = {}
    temp2 = {}
    for key_point in image_features[0]:
        temp[key_point.pt] = -1
        temp2[key_point.pt] = True
    feature_key_sequence.append(temp)
    feature_isBackground_sequence.append(temp2)

current_feature_index = 0
for i in range(len(image_sequence)):
    current_image = image_sequence[i]
    current_key, current_descriptor = image_sift_sequence[i]
    for next_frame_index in range(i, len(image_sequence)):
        next_image = image_sequence[next_frame_index]
        next_key, next_descriptor = image_sift_sequence[next_frame_index]
        current_labels = image_label_sequence[i]
        next_labels = image_label_sequence[next_frame_index]

        matrix = get_trans_matrices(current_key, current_descriptor, next_key, next_descriptor)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(current_descriptor, next_descriptor, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        for m in good:
            key_point1 = current_key[m.queryIdx].pt
            key_point1_transformed = get_new_coord(key_point1[0], key_point1[1], matrix)
            key_point2 = next_key[m.trainIdx].pt

            if feature_key_sequence[i][key_point1] == -1:
                feature_key_sequence[i][key_point1] = current_feature_index
                feature_key_sequence[next_frame_index][key_point2] = current_feature_index
                current_feature_index += 1

                x1, y1 = int(key_point1[0]), int(key_point1[1])
                x2, y2 = int(key_point2[0]), int(key_point2[1])

                for current_seg in current_labels:
                    for next_seg in next_labels:
                        # print(current_seg)
                        # print(current_seg.shape)
                        # print(x1, y1)
                        mask1 = np.array(current_seg == current_seg[y1][x1], dtype=np.uint8)
                        mask2 = np.array(next_seg == next_seg[y2][x2], dtype=np.uint8)
                        kp1, des1 = sift.detectAndCompute(current_image, mask=mask1)
                        kp2, des2 = sift.detectAndCompute(next_image, mask=mask2)

                        seg_matrix = get_trans_matrices(kp1, des1, kp2, des2)
                        key_point1_transformed_seg = get_new_coord(key_point1[0], key_point1[1], seg_matrix)
                        print(matrix, seg_matrix, sep='\n')
                        print(key_point1_transformed, key_point1_transformed_seg)
                        print(distance.euclidean(key_point1_transformed, key_point1_transformed_seg))


