import numpy as np
import cv2
from matplotlib import pyplot as plt
from segmentation import segmentation, mark_boundaries
import copy


image_sequence = []
# read cars2
path = "/Users/yifeima/Documents/CMPUT414/BackgroundSubtractionForMovingCamera/moseg_dataset/"
cap = cv2.VideoCapture(path + "/cars4/cars4_%02d.jpg")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image_sequence.append(frame)

for frame in image_sequence:
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


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
    out_image = image
    # img = cv2.drawKeypoints(image, kp, out_image)
    # cv2.imshow("features", img)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break


image_label_sequence = []
for image in image_sequence:
    labels = segmentation(image, 3)
    image_label_sequence.append(labels)

np.save('segmentation.npy', np.asarray(image_label_sequence))

image_label_sequence = np.load("segmentation.npy")
outputs = []
for i in range(len(image_sequence)):
    image = image_sequence[i]
    labels = image_label_sequence[i]
    for label in labels:
        outputs.append(mark_boundaries(image, label))

for img in outputs:
    cv2.imshow("segmentation", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


label = image_label_sequence[0][2]
image = image_sequence[0]
print(np.max(label))
mask = label == 3
mask = np.array(mask, dtype=np.uint8)

sift = cv2.xfeatures2d.SIFT_create(nfeatures=10000)
kp, des = sift.detectAndCompute(image, mask=mask)
out_image = image
img = cv2.drawKeypoints(image, kp, out_image)
print(len(kp))
cv2.imshow("sift", img)
cv2.waitKey(10000)


def get_trans_matrices(image1, pt1, image2, pt2):
    pass


# original_features_sequence = copy.deepcopy(image_sift_sequence)

for i in range(len(image_sequence)):
    image = image_sequence[i]
    current_key, current_descriptor = image_sift_sequence[i]

    for next_frame_index in range(i, len(image_sequence)):
        next_key, next_descriptor = image_sift_sequence[next_frame_index]

