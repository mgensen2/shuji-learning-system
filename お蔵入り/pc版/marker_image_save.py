import cv2
import numpy as np
aruco = cv2.aruco
p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
if __name__ == '__main__':

    marker =  [0] * 5 # 初期化
    for i in range(len(marker)):
        marker[i] = aruco.generateImageMarker(p_dict, i, 75) # 75x75 px
        cv2.imwrite(f'marker{i}.png', marker[i])