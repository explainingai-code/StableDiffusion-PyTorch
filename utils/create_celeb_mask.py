"""
This is exact copy of https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py
Only the folder locations are modified.
"""

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = 'data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
folder_save = 'data/CelebAMask-HQ/CelebAMask-HQ-mask'
img_num = 30000

if not os.path.exists(folder_save):
    os.mkdir(folder_save)

for k in tqdm(range(img_num)):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if os.path.exists(filename):
            im = cv2.imread(filename)
            im = im[:, :, 0]
            im_base[im != 0] = (idx + 1)

    filename_save = os.path.join(folder_save, str(k) + '.png')
    cv2.imwrite(filename_save, im_base)
