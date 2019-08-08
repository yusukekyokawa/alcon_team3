import cv2
import numpy as np
import glob
import pandas as pd
import os

def split(img):
    img = cv2.resize(img,(100,150))

    img1 = img[0:50,0:100]

    img2 = img[50:100,0:100]

    img3 = img[100:150,0:100]
    return img1, img2, img3

if __name__ == "__main__":
    path = "./train/imgs"
    csv_path = "./train/annotations.csv"

    df = pd.read_csv(csv_path)
    id_list = df("ID")
    uni1_list = df("Unicode1")
    uni2_list = df("Unicode2")
    uni3_list = df("Unicode3")

    #画像のフォルダを読み込む
    imglist = sorted(glob.glob(path + "/*"))
    #csvファイルを読み込む
    #print(imglist)

    save_dir = "./save_split"
    for idx, (img_path, uni1, uni2, uni3) in enumerate(zip(imglist, uni1_list, uni2_list, uni3_list)):
        save_path = save_dir + "/" + str(idx)
        os.makedirs(save_rath, exist_ok = True)

    #画像を読み込む
    img = cv2.imread(img_path)

    #3等分する
    img1, img2, img3 = split(img)

    #img1の保存
    cv2.imwrite(save_path+"/"+uni1+".jpg", img1)

    #img2の保存
    cv2.imwrite(save_path+"/"+uni2+".jpg", img2)

    #img3の保存
    cv2.imwrite(save_path+"/"+uni3+".jpg", img3)
    print(idx)

