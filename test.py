import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import *
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import *
from scipy.spatial import distance
from numpy import array, cross
from numpy.linalg import solve, norm
import os
import cv2
from skimage.segmentation import slic, mark_boundaries
import matplotlib.cm as cm


def get_trans_matrices(current_keys, current_descriptors, next_keys, next_descriptors):
    """
    Get the homogenous transformation matrix between two frame with key points.
    :param current_keys: key points of current frame
    :param current_descriptors: descriptors corresponding to the key points of current frame
    :param next_keys: key points of next frame
    :param next_descriptors: descriptors corresponding to the key points of next frame
    :return: the transformation matrix if able to get, else an array of 1
    """
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
    """
    Get the transformation matrix for each feature point(SIFT/SURF features) with its neighbouring feature points
    :param feature_distance:
    :param num_of_neighbour:
    :param kp_coords:
    :param current_features:
    :param next_features:
    :return: the transformation matrix if able to get, else an array of 1
    """
    idx = np.argpartition(feature_distance, num_of_neighbour)
    partial_kp = []
    for index in idx[:num_of_neighbour]:
        partial_kp.append(current_features[0][index])
    partial_des = current_features[1][idx[:num_of_neighbour]]
    return get_trans_matrices(partial_kp, partial_des, next_features[0], next_features[1])


def get_mask(image, kp):
    """
    Returns the binary mask using SLIC
    :param image:
    :param kp: which is the keypoints of the foreground
    :return: a ndarray of binary mask
    """
    labels = slic(image, 500, compactness=10.0, max_iter=20, sigma=1, enforce_connectivity=True,
                  start_label=1)
    region_list = np.zeros(np.max(labels))
    for keypoint in kp:
        coords = np.flip(np.asarray(keypoint.pt).astype(int))
        region_index = labels[coords[0]][coords[1]]
        region_list[region_index] += 1
    # print(region_list)
    mask = np.zeros(labels.shape)
    for index in range(len(region_list)):
        if region_list[index] > 4:
            mask[np.where(labels == index)] = 1
    # print(mask)
    # plt.imsave("mask_test.png", mask, cmap=cm.gray)
    return mask


def display(model):
    """
    Get the predicted result, and save the result to corresponding folders
    :param model:
    :return:
    """
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

        # print(len(current_matrices), len(current_features[0]), current_index)

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
        # print(labels.shape)
        # print(np.sum(labels))

        kp = []
        for i in range(len(labels)):
            if labels[i] != 0:
                kp.append(current_features[0][i])

        image = image_sequence[current_index]
        mask = get_mask(image, kp)
        plt.imsave("/Users/yifeima/Documents/CMPUT414/BackgroundSubtractionForMovingCamera/mask/mask_{}.png"
                   .format(current_index),
                   mask, cmap=cm.gray)
        out_image = image

        img = cv2.drawKeypoints(image, kp, out_image)
        cv2.imwrite("/Users/yifeima/Documents/CMPUT414/BackgroundSubtractionForMovingCamera"
                    "/output/output_{}.jpg".format(current_index), img)
        # exit()
        # cv2.imshow("features", img)
        # cv2.waitKey(10)


def duplicate_data(matrices, ground_truth_sift_label):
    """
    Duplicate data for training, returns a randomized training data with evenly amount for foreground and background.
    :param matrices:
    :param ground_truth_sift_label:
    :return: An evenly distributed training data
    """
    indices = np.where(ground_truth_sift_label == 1)[0]
    matrices_foreground = matrices[indices]
    matrices_foreground = np.resize(matrices_foreground,
                                    (len(ground_truth_sift_label) - 2*sum(ground_truth_sift_label), 3, 3))
    labels = np.ones(len(ground_truth_sift_label) - 2*sum(ground_truth_sift_label))
    matrices = np.append(matrices, matrices_foreground, axis=0)
    ground_truth_sift_label = np.append(ground_truth_sift_label, labels)

    p = np.random.permutation(len(matrices))
    return matrices[p], ground_truth_sift_label[p]


def classifier(matrices, ground_truth_sift_label):
    """
    In this function, we tried the naive bayes approach and MLP classifier, turned out MLP classifier works very well.
    :param matrices: All the matrices for training
    :param ground_truth_sift_label: the ground truth label for the matrices
    :return:
    """
    matrices, ground_truth_sift_label = duplicate_data(matrices, ground_truth_sift_label)
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

    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(100, 40, 10),
                          batch_size=100,
                          learning_rate='adaptive', learning_rate_init=0.001,
                          alpha=0.0001, max_iter=10000000000,
                          early_stopping=True,
                          n_iter_no_change=100)
    # model = BernoulliNB()
    model.fit(matrix_features, ground_truth_sift_label)

    plt.plot(model.loss_curve_)
    plt.show()
    plt.savefig('loss_curve.png')
    plt.clf()
    plt.plot(model.validation_scores_)
    plt.show()
    plt.savefig('validation_scores.png')

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

    print(model.score(matrix_features, ground_truth_sift_label))
    print("results for frame 10")
    print(model.score(matrix_features_test, ground_truth_test))
    predict = model.predict(matrix_features_test)
    print("Total number of feature points {}".format(len(ground_truth_test)))
    print("Total number of foreground feature points {}".format(np.sum(ground_truth_test)))
    print("Total number of predicted foreground feature points {}".format(np.sum(predict)))

    print("Confusion matrix:\n", confusion_matrix(ground_truth_test, predict))

    display(model)


def matrix_to_point(trans_matrix):
    """
    In this function, we tried to use SVD to map the matrix to 3D vector space.
    :param trans_matrix:
    :return:
    """
    u, s, _ = np.linalg.svd(trans_matrix)
    u_s_sum = np.dot(u, np.diag(s)).sum(axis=1)
    return u_s_sum


def matrix_processing(matrices):
    """
    Get all the result by applying SVD
    :param matrices:
    :return:
    """
    matrix_features = []
    for matrix in matrices:
        feature_point = matrix_to_point(matrix.astype(float))
        matrix_features.append(feature_point)
    matrix_features = np.asarray(matrix_features)
    return matrix_features


def matrix_processing_tsne(matrices):
    """
    This function tries different dimensionality reduction method to map a matrix to 3D vector space.
    :param matrices:
    :return:
    """
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
    """
    This function calculate the angle between 3 vector a, b, c
    Our idea was to calculate the angel between each vector to get a new vector, proved not working since each unit
    vector is perpendicular to each other.
    :param a:
    :param b:
    :param c:
    :return: a list contains angle between each vector
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    a_b = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    a_c = np.arccos(np.clip(np.dot(a, c), -1.0, 1.0))
    b_c = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
    return [a_b, a_c, b_c]


def matrix_processing_new(matrices):
    """
    This function is not currently used in our approach, it was used to try different method, this function currently
    append all the matrices of a frame then calculate SVD of the 3X3N matrix, where N is the number of total matrices.
    :param matrices:
    :return: a array contains vectorized matrices
    """
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
    """
    Main function of our approach, read data from *.npy file, then tries different approach for test
     choose one of our approach to test (classifier, svd, tsne, etc.)
    :return:
    """
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
    classifier(matrices, ground_truth_sift_label)
    exit()

    '''
        the following code is used to plot vectors in 3D space, is not part of main algorithm
    '''

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
