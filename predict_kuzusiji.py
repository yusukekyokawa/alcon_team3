import glob
import cv2
from keras.models import load_model
import os
from natsort import natsorted
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd


label_dict = {0:"U+3042", 1:"U+3044", 2:"U+3046", 3:"U+3048", 4:"U+304A", 5:"U+304B", 6:"U+304D",
        7:"U+304F", 8:"U+3051", 9:"U+3053", 10:"U+3055", 11:"U+3057", 12:"U+3059",13:"U+305B",
        14:"U+305D", 15:"U+305F", 16:"U+3061", 17:"U+3064", 18:"U+3066", 19:"U+3068", 20:"U+306A",
        21:"U+306B", 22:"U+306C", 23:"U+306D", 24:"U+306E", 25:"U+306F", 26:"U+3072", 27:"U+3075",
        28:"U+3078", 29:"U+307B", 30:"U+307E", 31:"U+307F", 32:"U+3080", 33:"U+3081", 34:"U+3082",
        35:"U+3084", 36:"U+3086", 37:"U+3088", 38:"U+3089", 39:"U+308A", 40:"U+308B", 41:"U+308C",
        42:"U+308D", 43:"U+308F", 44:"U+3090", 45:"U+3091", 46:"U+3092", 47:"U+3093"
}



def split(img):
    """
    入力した画像を100*150にリサイズし、
    画像を縦方向に3等分する
    """
    img = cv2.resize(img,(64,192))

    img1 = img[0:64,0:64]

    img2 = img[64:128,0:64]

    img3 = img[128:192,0:64]
    return img1, img2, img3




if __name__ == "__main__":
    test_img_dir = "../alcon2019/dataset/test/imgs"
    # test画像読み込みリストで
    test_img_list = natsorted(glob.glob(test_img_dir + "/*"))

    # testデータのcsvファイルを作成
    with open("test_prediction.csv", "w") as f:
        f.write("ID,Unicode1,Unicode2,Unicode3" + "\n")


    

    # CNNモデル読み込み
    model = load_model("./a.h5")

    # 保存先をssdに
    ssd = "/media/mizuno/8b052113-9e75-4fb4-af54-53f93bbe44a7"


    # 切り出した画像の保存rootを
    save_root = os.path.join(ssd, "alcon","split_test_gray")
    # for文で画像を一枚ずつ取り出す。
    for index, test_img_path in enumerate(test_img_list):
        
        print("ID: {} Path: {}".format(index, test_img_path))

        # # 画像の読み込み
        test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        
        # # 画像のリサイズと切り出し
        uni1_img, uni2_img, uni3_img = split(test_img)
        
        # #################################
        # # 保存するディレクトリの作成
        # save_split_dir = os.path.join(save_root, str(index))
        # os.makedirs(save_split_dir, exist_ok=True)
        
        # # 切り出した画像を保存する
        # uni1_save_path = os.path.join(save_split_dir, "uni1.jpg")
        # uni2_save_path = os.path.join(save_split_dir, "uni2.jpg")
        # uni3_save_path = os.path.join(save_split_dir, "uni3.jpg")

        # cv2.imwrite(uni1_save_path, uni1_img)
        # cv2.imwrite(uni2_save_path, uni2_img)
        # cv2.imwrite(uni3_save_path, uni3_img)
        ########################################


        # 画像の正規化
        uni1_img = uni1_img.astype('float32')/255.0
        uni2_img = uni2_img.astype('float32')/255.0
        uni3_img = uni3_img.astype('float32')/255.0

        # 画像を4次元配列に
        uni1_img = uni1_img[..., None]
        uni2_img = uni2_img[..., None]
        uni3_img = uni3_img[..., None]

        uni1_img = uni1_img[None, ...]
        uni2_img = uni2_img[None, ...]
        uni3_img = uni3_img[None, ...]

        # CNNモデルで予測48クラスの中で一番高いラベルを出力
        uni1_prediction = np.argmax(model.predict(uni1_img))

        uni2_prediction = np.argmax(model.predict(uni2_img))

        uni3_prediction = np.argmax(model.predict(uni3_img))

        # ラベルとunicodeに変換
        uni1_result = label_dict[uni1_prediction]
        uni2_result = label_dict[uni2_prediction]
        uni3_result = label_dict[uni3_prediction]

        # csvファイルに書き込み
        with open("test_prediction.csv", "a") as f:
            f.write(str(index) + "," + uni1_result + "," + uni2_result + "," + uni3_result + "\n")