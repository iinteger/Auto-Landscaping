# 이미지체서 엣지를 추출해 저장
# canny threshold는 heuristic하게 결정함

import os
import cv2
import glob
import random

dataset = "twilight landscape"  # data folder

color_path = "images/"+dataset+"/"
image_ids = glob.glob(color_path+"*")
path = "images/"+dataset+" edge/"

for i in image_ids:
    try:  # 간혹 load가 안되는 경우가 있음
        canny = ~cv2.Canny(cv2.imread(i), random.randrange(100, 201), random.randrange(500, 601))
    except:
        os.remove(i)
        continue
    cv2.imwrite(path+i[len(color_path):], canny)