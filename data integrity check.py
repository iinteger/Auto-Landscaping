# 간혹 data load가 안될때가 있음
# 먼저 무결성 체크
from PIL import Image
import cv2
import glob
import PIL

target_path = "images/train/twilight landscape/"
image_ids = glob.glob(target_path+"*")

for i in image_ids:
    try:
        color_image = Image.open(i).convert("RGB")
    except PIL.UnidentifiedImageError as e:
        print(e)