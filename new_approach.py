import numpy as np
import cv2
from matplotlib import pyplot as plt
from segmentation import segmentation, mark_boundaries
from scipy.spatial import distance

from sklearn.cluster import KMeans, DBSCAN

from mpl_toolkits.mplot3d import Axes3D

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
ground_truth_image = cv2.imread(path + "/cars4/cars4_10.pgm")


image_sift_sequence = []

sift = cv2.xfeatures2d.SURF_create()
count = 0
for image in image_sequence:
    kp, des = sift.detectAndCompute(image, None)
    image_sift_sequence.append((kp, des))
    # print(len(kp))
    out_image = image
    img = cv2.drawKeypoints(image, kp, out_image)
    cv2.imwrite("/Users/yifeima/Documents/CMPUT414/BackgroundSubtractionForMovingCamera"
                "/feature image/output_sift_{}.jpg".format(count), img)
    count += 1
    # cv2.imshow("features", img)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break
exit()

# image_label_sequence = []
# for image in image_sequence:
#     labels = segmentation(image, 3)
#     image_label_sequence.append(labels)
#
# np.save('segmentation.npy', np.asarray(image_label_sequence))

image_label_sequence = np.load("segmentation.npy")
# outputs = []
# for i in range(len(image_sequence)):
#     image = image_sequence[i]
#     labels = image_label_sequence[i]
#     # j = 0
#     for label in labels:
#         outputs.append(mark_boundaries(image, label))
#         # cv2.imwrite('segs_{}_{}.jpg'.format(i, j), mark_boundaries(image, label))
#         # j += 1


# # i = 0
# for img in outputs:
#     # img = cv2.convertScaleAbs(img, alpha=(255.0))
#     # cv2.imwrite('segs_{}.jpg'.format(i), img, )
#     cv2.imshow("segmentation", img)
#     # i += 1
#     # if i >= 10: exit()
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break


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

    # src_pts = np.float32([current_keys[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # dst_pts = np.float32([next_keys[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #
    # M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # print(len(good))
    if len(good) > 5:
        src_pts = np.float32([current_keys[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # print(good)
        # for pts in src_pts:
        #     print(pts)
        # print(len(src_pts))
        dst_pts = np.float32([next_keys[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        M, _ = cv2.findHomography(src_pts, dst_pts, 0)

        return M
    else:
        return np.zeros(1)


def get_new_coord(x, y, matrix):
    p = (x, y)  # original point
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    p_after = (int(px), int(py))  # after transformation
    return p_after

# image_dict_sequence = []
# for kp, des in image_sift_sequence:
#     temp = {}
#     for keypoint in kp:
#         temp[]


def get_feature_matrix():
    idx = np.argpartition(feature_distance, num_of_neighbour)
    neighbour_coords = np.asarray(kp_coords)[idx[:num_of_neighbour]]
    neighbour_coords_index = []
    partial_kp = []
    for index in idx[:num_of_neighbour]:
        partial_kp.append(current_features[0][index])
    partial_des = current_features[1][idx[:num_of_neighbour]]
    return get_trans_matrices(partial_kp, partial_des, next_features[0], next_features[1])


for current_index in range(len(image_sequence) - 1):
    current_features = image_sift_sequence[10]
    next_features = image_sift_sequence[10+3]
    kp_index = {}
    kp_coords = []
    # coord_to_kp_index = {}
    ground_truth_sift_label = []
    for i in range(len(current_features[0])):
        keypoint = current_features[0][i]
        kp_index[keypoint] = i
        kp_coords.append(keypoint.pt)
        # coord_to_kp_index[keypoint.pt] = i
        coords = np.flip(np.asarray(keypoint.pt).astype(int))
        if ground_truth_image[coords[0]][coords[1]].all() == 0:
            ground_truth_sift_label.append(0)
        else:
            ground_truth_sift_label.append(1)
    ground_truth_sift_label = np.asarray(ground_truth_sift_label)

    matrix_features = []
    matrix_feature_to_index = {}

    trans_matrices_for_store = []
    for i in range(len(current_features[0])):
        keypoint = current_features[0][i]
        feature_distance = distance.cdist(np.asarray([keypoint.pt]), np.asarray(kp_coords)).squeeze()
        num_of_neighbour = 15
        # idx = np.argpartition(feature_distance, num_of_neighbour)
        # neighbour_coords = np.asarray(kp_coords)[idx[:num_of_neighbour]]
        # neighbour_coords_index = []
        # partial_kp = []
        # for index in idx[:num_of_neighbour]:
        #     partial_kp.append(current_features[0][index])
        # partial_des = current_features[1][idx[:num_of_neighbour]]
        # trans_matrix = get_trans_matrices(partial_kp, partial_des, next_features[0], next_features[1])
        trans_matrix = get_feature_matrix()
        while True:
            if trans_matrix.all() != np.zeros(1):
                break
            num_of_neighbour += 5
            trans_matrix = get_feature_matrix()

        trans_matrices_for_store.append(trans_matrix)
        u, s, _ = np.linalg.svd(trans_matrix)
        # print(u, np.diag(s), _)
        # print(np.dot(u, np.diag(s)))
        u_s_sum = np.dot(u, np.diag(s)).sum(axis=1)
        matrix_features.append(u_s_sum)
        # matrix_feature_to_index[u_s_sum] = i

    trans_matrices_for_store = np.asarray(trans_matrices_for_store, dtype=object)
    np.save("test.npy", trans_matrices_for_store)
    np.save("test_ground.npy", ground_truth_sift_label)
    exit()

    matrix_features = np.asarray(matrix_features)

    indices_test_labels_z = np.where(matrix_features[:, 0] < -5000)[0]

    # clustering = KMeans(n_clusters=2, random_state=0).fit(matrix_features)
    # clustering = DBSCAN(eps=1, min_samples=5, n_jobs=-1).fit(matrix_features)
    # feature_labels = clustering.labels_

    indices_of_label = np.where(ground_truth_sift_label == 0)[0]
    indices_of_label2 = np.where(ground_truth_sift_label == 1)[0]

    matrix_features1 = matrix_features[indices_of_label]
    matrix_features2 = matrix_features[indices_of_label2]

    plots = []
    for item in matrix_features:
        if item[0] > -20000:
            plots.append(item)
    plots = np.asarray(plots)

    plots1 = []
    plots2 = []
    for item in matrix_features1:
        if item[0] > -20000:
            plots1.append(item)
    plots1 = np.asarray(plots1)

    for item in matrix_features2:
        if item[0] > -20000:
            plots2.append(item)
    plots2 = np.asarray(plots2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(plots[:, 0], plots[:, 1], plots[:, 2], color="blue")
    ax.scatter3D(plots1[:, 0], plots1[:, 1], plots1[:, 2], color="green")
    ax.scatter3D(plots2[:, 0], plots2[:, 1], plots2[:, 2], color="red")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.ion()
    plt.show()

    kp = []
    for index in indices_of_label2:
        kp.append(current_features[0][index])

    image = image_sequence[current_index]
    out_image = image

    img = cv2.drawKeypoints(image, kp, out_image)
    cv2.imshow("features", img)
    cv2.waitKey(0)
    # if cv2.waitKey(1000000) & 0xFF == ord('q'):
    #     break

    break
