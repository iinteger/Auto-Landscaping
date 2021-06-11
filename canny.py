# 이미지를 읽어와 엣지로 저장

import cv2

import matplotlib.pyplot as plt


import glob
import random

dataset = "twilight landscape"

target_path = "images/train/"+dataset+"/"
image_ids = glob.glob(target_path+"*")
path = "images/train/"+dataset+"/"

for i in image_ids:
    print(i)
    image = cv2.imread(i)
    print(type(image))
    canny = ~cv2.Canny(image, random.randrange(100, 151), random.randrange(500, 551))
    cv2.imwrite(path+i[len(target_path):], canny)