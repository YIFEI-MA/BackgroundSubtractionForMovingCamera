import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import *

from numpy import array, cross
from numpy.linalg import solve, norm


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
    # X_embedded = TSNE(n_components=3, perplexity=10, n_iter=1000, metric="sqeuclidean", random_state=0,
    #                   method='exact', n_jobs=-1).fit_transform(matrix_features)
    # X_embedded = Isomap(5, 3).fit_transform(matrix_features)
    # X_embedded = MDS(3, max_iter=100, n_init=1).fit_transform(matrix_features)
    X_embedded = SpectralEmbedding(n_components=3, n_neighbors=10).fit_transform(matrix_features)
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
    a = np.dot(matrices[1].astype(float), u[:, 0])
    b = np.dot(matrices[1].astype(float), u[:, 1])
    c = np.dot(matrices[1].astype(float), u[:, 2])

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
    matrices = np.load("matrices.npy", allow_pickle=True)
    print(len(matrices))
    ground_truth_sift_label = np.load("ground_truth.npy")

    matrix_features = matrix_processing_new(matrices)
    # matrix_features = matrix_processing_tsne(matrices)
    # exit()
    #
    # matrix_features = matrix_processing(matrices)
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
