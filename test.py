import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import *
from sklearn.naive_bayes import *
from scipy.spatial import distance
from numpy import array, cross
from numpy.linalg import solve, norm
import os
import cv2


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


def get_feature_matrix(feature_distance, num_of_neighbour, kp_coords, current_features, next_features):
    idx = np.argpartition(feature_distance, num_of_neighbour)
    partial_kp = []
    for index in idx[:num_of_neighbour]:
        partial_kp.append(current_features[0][index])
    partial_des = current_features[1][idx[:num_of_neighbour]]
    return get_trans_matrices(partial_kp, partial_des, next_features[0], next_features[1])


def display(model):
    image_sequence = []
    # read cars2
    path = "/Users/yifeima/Documents/CMPUT414/BackgroundSubtractionForMovingCamera/moseg_dataset/"
    cap = cv2.VideoCapture(path + "/cars4/cars4_%02d.jpg")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_sequence.append(frame)

    image_sift_sequence = []

    sift = cv2.xfeatures2d.SURF_create()
    for image in image_sequence:
        kp, des = sift.detectAndCompute(image, None)
        image_sift_sequence.append((kp, des))

    for current_index in range(len(image_sequence) - 1):
        current_matrices = []
        current_features = image_sift_sequence[current_index]
        next_features = image_sift_sequence[current_index + 1]
        kp_index = {}
        kp_coords = []
        ground_truth_sift_label = []
        for i in range(len(current_features[0])):
            keypoint = current_features[0][i]
            kp_index[keypoint] = i
            kp_coords.append(keypoint.pt)

        for i in range(len(current_features[0])):
            keypoint = current_features[0][i]
            feature_distance = distance.cdist(np.asarray([keypoint.pt]), np.asarray(kp_coords)).squeeze()
            num_of_neighbour = 15
            trans_matrix = get_feature_matrix(feature_distance, num_of_neighbour, kp_coords,
                                              current_features, next_features)
            while True:
                if trans_matrix.all() != np.zeros(1):
                    break
                num_of_neighbour += 5
                trans_matrix = get_feature_matrix(feature_distance, num_of_neighbour, kp_coords,
                                                  current_features, next_features)
            current_matrices.append(trans_matrix)

        print(len(current_matrices), len(current_features[0]), current_index)

        matrix_features_test = []
        for matrix in current_matrices:
            features = np.squeeze(np.matrix.flatten(matrix))
            # w, _ = np.linalg.eig(matrix)
            u, s, _ = np.linalg.svd(matrix)
            u_s = np.dot(u, np.diag(s))
            features = np.concatenate((features, s, u_s[:, 0], u_s[:, 1], u_s[:, 2]))
            matrix_features_test.append(features)
        matrix_features_test = np.asarray(matrix_features_test)

        labels = model.predict(matrix_features_test)
        print(labels.shape)
        print(np.sum(labels))

        kp = []
        for i in range(len(labels)):
            if labels[i] != 0:
                kp.append(current_features[0][i])

        image = image_sequence[current_index]
        out_image = image

        img = cv2.drawKeypoints(image, kp, out_image)
        cv2.imwrite("output_{}.jpg".format(current_index), img)
        cv2.imshow("features", img)
        cv2.waitKey(0)


    pass


def naive_bayes(matrices, ground_truth_sift_label):
    matrices = matrices.astype(float)
    matrix_features = []
    for matrix in matrices:
        features = np.squeeze(np.matrix.flatten(matrix))
        # w, _ = np.linalg.eig(matrix)
        u, s, _ = np.linalg.svd(matrix)
        u_s = np.dot(u, np.diag(s))
        features = np.concatenate((features, s, u_s[:, 0], u_s[:, 1], u_s[:, 2]))
        matrix_features.append(features)
    matrix_features = np.asarray(matrix_features)
    model = BernoulliNB()
    model.fit(matrix_features, ground_truth_sift_label)

    matrices_test = np.load("test.npy", allow_pickle=True)
    ground_truth_test = np.load("test_ground.npy")

    matrix_features_test = []
    for matrix in matrices_test:
        matrix = matrix.astype(float)
        features = np.squeeze(np.matrix.flatten(matrix))
        # w, _ = np.linalg.eig(matrix)
        u, s, _ = np.linalg.svd(matrix)
        u_s = np.dot(u, np.diag(s))
        features = np.concatenate((features, s, u_s[:, 0], u_s[:, 1], u_s[:, 2]))
        matrix_features_test.append(features)
    matrix_features_test = np.asarray(matrix_features_test)

    print("XXXXX")
    print(model.score(matrix_features_test, ground_truth_test))
    print(np.sum(model.predict(matrix_features_test)))
    print(np.sum(ground_truth_test))
    print(len(ground_truth_test))

    display(model)

    pass


def matrix_to_point(trans_matrix):
    u, s, _ = np.linalg.svd(trans_matrix)
    u_s_sum = np.dot(u, np.diag(s)).sum(axis=1)
    return u_s_sum


def matrix_processing(matrices):
    matrix_features = []
    for matrix in matrices:
        feature_point = matrix_to_point(matrix.astype(float))
        matrix_features.append(feature_point)
    matrix_features = np.asarray(matrix_features)
    return matrix_features


def matrix_processing_tsne(matrices):
    matrices = matrices.astype(float)
    matrix_features = []
    for matrix in matrices:
        matrix_features.append(np.squeeze(np.matrix.flatten(matrix)))
    """
    ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation',
     'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
     'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean',
     'haversine'], or 'precomputed', or a callable
    """
    X_embedded = TSNE(n_components=3, perplexity=10, n_iter=1000, metric="l2", random_state=0,
                      method='exact', n_jobs=-1).fit_transform(matrix_features)
    # X_embedded = Isomap(5, 3).fit_transform(matrix_features)
    # X_embedded = MDS(3, max_iter=100, n_init=1).fit_transform(matrix_features)
    # X_embedded = SpectralEmbedding(n_components=3, n_neighbors=10).fit_transform(matrix_features)
    return X_embedded


def get_angle(a, b, c):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    a_b = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    a_c = np.arccos(np.clip(np.dot(a, c), -1.0, 1.0))
    b_c = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
    return [a_b, a_c, b_c]

    pass


def matrix_processing_new(matrices):
    concatenated_matrices = matrices[0].astype(float)
    print(len(concatenated_matrices))
    matrix_features = []
    for index in range(1, len(matrices)):
        concatenated_matrices = np.concatenate((concatenated_matrices, matrices[index].astype(float)), axis=1)
    print(concatenated_matrices.shape)
    u, s, _ = np.linalg.svd(concatenated_matrices)
    # a = np.dot(matrices[1].astype(float), u[:, 0])
    # b = np.dot(matrices[1].astype(float), u[:, 1])
    # c = np.dot(matrices[1].astype(float), u[:, 2])

    for matrix in matrices:
        a = np.dot(matrix.astype(float), u[:, 0])
        b = np.dot(matrix.astype(float), u[:, 1])
        c = np.dot(matrix.astype(float), u[:, 2])
        matrix_features.append(get_angle(a, b, c))

    return np.asarray(matrix_features)

    # print(a, b, c)
    # plots1 = [a, b, c]
    # plots1 = np.asarray(plots1)
    #
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # # ax.scatter3D(plots[:, 0], plots[:, 1], plots[:, 2], color="blue")
    # # ax.scatter3D(plots1[:, 0], plots1[:, 1], plots1[:, 2], color="green")
    # #
    # # ax.set_xlabel('X Label')
    # # ax.set_ylabel('Y Label')
    # # ax.set_zlabel('Z Label')
    # #
    # # plt.ion()
    # # plt.show()

    # origin = [0, 0, 0]
    # X, Y, Z = zip(origin, origin, origin)
    # # X, Y, Z = zip(u[:, 0], u[:, 1], u[:, 2])
    # U, V, W = zip(a, b, c)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.01)
    # plt.show()


def testing():
    path = "/Users/yifeima/Documents/CMPUT414/BackgroundSubtractionForMovingCamera/"
    matrices_file = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and 'matrices' in i]
    ground_truth_file = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and 'ground_truth' in i]

    matrices_temp = []
    for file in matrices_file:
        matrices_temp.append(np.load(file, allow_pickle=True))
    matrices_temp = np.asarray(matrices_temp)
    matrices = np.array(matrices_temp[0].astype(float))
    for index in range(1, len(matrices_temp)):
        matrices = np.concatenate((matrices, matrices_temp[index].astype(float)))

    ground_temp = []
    for file in ground_truth_file:
        ground_temp.append(np.load(file, allow_pickle=True))
    ground_temp = np.asarray(ground_temp)
    ground_truth_sift_label = np.array(ground_temp[0])
    for index in range(1, len(ground_temp)):
        ground_truth_sift_label = np.concatenate((ground_truth_sift_label, ground_temp[index]))

    # matrices = np.load("matrices.npy", allow_pickle=True)
    # print(len(matrices))
    # ground_truth_sift_label = np.load("ground_truth.npy")
    # print(ground_truth_sift_label)

    # matrix_features = matrix_processing_new(matrices)
    # matrix_features = matrix_processing_tsne(matrices)
    # exit()
    #
    naive_bayes(matrices, ground_truth_sift_label)
    exit()
    matrix_features = matrix_processing(matrices)
    #
    indices_of_label = np.where(ground_truth_sift_label == 0)[0]
    indices_of_label2 = np.where(ground_truth_sift_label == 1)[0]

    matrix_features1 = matrix_features[indices_of_label]
    matrix_features2 = matrix_features[indices_of_label2]

    plots1 = []
    plots2 = []
    for item in matrix_features1:
        plots1.append(item)
        # if item[0] < 0:
        #     plots1.append(item)
    plots1 = np.asarray(plots1)

    for item in matrix_features2:
        plots2.append(item)
        # if item[0] < 0:
        #     plots2.append(item)
    plots2 = np.asarray(plots2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(plots[:, 0], plots[:, 1], plots[:, 2], color="blue")
    ax.scatter3D(plots1[:, 0], plots1[:, 1], plots1[:, 2], color="green", marker="o")
    ax.scatter3D(plots2[:, 0], plots2[:, 1], plots2[:, 2], color="red", marker="^")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.ion()
    plt.show()


testing()
