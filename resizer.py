import cv2
from pathlib import Path
from itertools import chain

dir = Path("C:/Users/giova/Desktop/Poli/Magistrale/ML/Progetto/Datasets/High_Res/train")

img_list = []
img_list.extend(chain(dir.glob("*/*/*.png")))
n_img = len(img_list)

idx = 1
for img in img_list:
    cv_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)  # Keeps alpha channel if present
    h, w, _ = cv_img.shape

    if h%4 == 0 and w%4 == 0:
        idx += 1
        continue

    h_new = h - h%4
    w_new = w - w%4
    resized_img = cv_img[0:h_new, 0:w_new]

    cv2.imwrite(img, resized_img)
    
    if idx%100==0:
        print(idx/n_img*100)
    idx += 1