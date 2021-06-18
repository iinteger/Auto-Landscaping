# 간혹 data load가 안될때가 있음. image download 도중 생기는 문제로 추측
# load가 안되는 이미지를 예와처리로 삭제

from PIL import Image
import glob
import PIL
import os

target_path = "images/beautiful landscape/"
image_ids = glob.glob(target_path+"*")
print(len(image_ids))

for i in image_ids:
    try:
        color_image = Image.open(i).convert("RGB")
    except PIL.UnidentifiedImageError as e:
        print(e)
        os.remove(i)
