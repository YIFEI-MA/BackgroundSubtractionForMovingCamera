import numpy as np
import cv2
from matplotlib import pyplot as plt


image_sequence = []
# read cars2
# for i in range(1, 31):
#     image_sequence.append()
path = "/Users/yifeima/Documents/CMPUT414/BackgroundStractionForMovingCamera/moseg_dataset/cars2/"
cap = cv2.VideoCapture(path + "cars2_%02d.jpg")
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

sift = cv2.SIFT_create()
for image in image_sequence:
    kp, des = sift.detectAndCompute(image, None)
    image_sift_sequence.append((kp, des))
    out_image = image
    img = cv2.drawKeypoints(image, kp, out_image)
    cv2.imshow("sift", img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


