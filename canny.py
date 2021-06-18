# 이미지체서 엣지를 추출해 저장, load가 안되는 이미지는 삭제
# canny threshold는 heuristic하게 결정함

import os
import cv2
import glob
import random

dataset = "twilight landscape"  # data folder

color_path = "images/"+dataset+"/"
image_ids = glob.glob(color_path+"*")
edge_path = "images/"+dataset+" edge/"

for i in image_ids:
    try:  # load가 안되는 이미지는 예외처리로 삭제
        canny = ~cv2.Canny(cv2.imread(i), random.randrange(100, 201), random.randrange(500, 601))
    except:
        os.remove(i)
        continue
    cv2.imwrite(edge_path+i[len(color_path):], canny)