import numpy as np
from matplotlib import pyplot as plt


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


def testing():
    matrices = np.load("matrices.npy", allow_pickle=True)
    ground_truth_sift_label = np.load("ground_truth.npy")
    matrix_features = matrix_processing(matrices)

    indices_of_label = np.where(ground_truth_sift_label == 0)[0]
    indices_of_label2 = np.where(ground_truth_sift_label == 1)[0]

    matrix_features1 = matrix_features[indices_of_label]
    matrix_features2 = matrix_features[indices_of_label2]

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


testing()
