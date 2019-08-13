from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam,SGD,Nadam
from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import os, glob
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pickle
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import unicode_to_kana
# データロード
ClassNum = 48 # クラス数
ImageData = [] # 画像を格納する配列
LabelData = [] # ラベルを格納する配列
# トレーニングデータがあるフォルダのパス
FolderPath = "../../alcon2019/dataset/train_kana/"
path = sorted(glob.glob(FolderPath + '**'))
print(len(path))

with open("label.csv", "w") as f:
	f.write("index,unicode,kana"+"\n")

Xtrain = []
Xtest = []
Ytrain = []
Ytest = []
for index, folder in enumerate(path):
	FolderName = os.path.basename(folder)
	# csvに書き出し
	with open("label.csv", "a") as f:
		f.write(str(index) + "," + FolderName + "," + unicode_to_kana(FolderName) +  "\n")
	ImagePath = sorted(glob.glob(folder + '/*.jpg'))
	
	print("FolderName is {}".format(FolderName))
	# サブフォルダーの中の画像枚数分繰り返す
	kanamoji_data = []
	kanamoji_label = []
	for ImageFile in ImagePath:
		"""
		img_to_array:PIL形式をndarrayに変換する
		load_img:画像を開く 
		target_sizeはリサイズする大きさ
		color_modeはRGB,grayscaleなど選べる
		"""
		# print(ImageFile)
		img = img_to_array(load_img(ImageFile, target_size=(64, 64), color_mode='grayscale'))
		
		# かな文字データに追加
		kanamoji_data.append(img)
		# かな文字ラベルに追加
		kanamoji_label.append(index)
	
	print("{}".format(unicode_to_kana(FolderName)))
	print("画像の総数は {}".format(len(kanamoji_data)))
	tmp_Xtrain, tmp_Xtest, tmp_Ytrain, tmp_Ytest = train_test_split(kanamoji_data, kanamoji_label, test_size=0.2, random_state=111)
	print("trainの枚数は {}".format(len(tmp_Xtrain)))
	print("testの枚数は {}".format(len(tmp_Xtest)))
	print("######################################")
	# データをtrainsize0.8でsplitしたものを共通ファイルに追加
	Xtrain.extend(tmp_Xtrain)
	Xtest.extend(tmp_Xtest)

	#　データをtestsize0.2でsplitしたものを共通ファイルに追加
	Ytrain.extend(tmp_Ytrain)
	Ytest.extend(tmp_Ytest)

		########################
		# 画像を格納
		# ImageData.append(img)
		# ラベルを格納
		# LabelData.append(index)
		########################

	# trainとtestに分ける。
	
# # 正規化
# ImageData = np.array(ImageData)
# LabelData = np.array(LabelData)
# ImageData = ImageData.astype('float32')
# ImageData = ImageData / 255.0
# LabelData = np_utils.to_categorical(LabelData, ClassNum)
# # 画像とラベルをtrain, validationと分ける
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(ImageData, LabelData, test_size=0.2, random_state=111)
# # データの保存
# ssd = '/media/mizuno/8b052113-9e75-4fb4-af54-53f93bbe44a7'
# with open(os.path.join(ssd,"Xtrain.pickle"), "wb") as f:
#         pickle.dump(Xtrain, f, protocol=4)
# with open(os.path.join(ssd,"Xtest.pickle"), "wb") as f:
#         pickle.dump(Xtest, f, protocol=4)
# with open(os.path.join(ssd,"Ytrain.pickle"), "wb") as f:
#         pickle.dump(Ytrain, f, protocol=4)
# with open(os.path.join(ssd,"Ytest.pickle"), "wb") as f:
#         pickle.dump(Ytest, f, protocol=4)


# datasetの読み込み
ssd = '/media/mizuno/8b052113-9e75-4fb4-af54-53f93bbe44a7'
Xtrain  = pickle.load(open(os.path.join(ssd,'Xtrain.pickle'),'rb'))
Xtest  = pickle.load(open(os.path.join(ssd,'Xtest.pickle'),'rb'))
Ytrain  = pickle.load(open(os.path.join(ssd,'Ytrain.pickle'),'rb'))
Ytest  = pickle.load(open(os.path.join(ssd,'Ytest.pickle'),'rb'))


# モデルの作成
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(Xtrain.shape[1:]),
		  activation='relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(ClassNum, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

history = model.fit(Xtrain, Ytrain, 
				batch_size=1024, epochs=10, 
				validation_data=(Xtest,Ytest),
				verbose=1)
model.save('./CNN_models/a.h5')
model = load_model('./CNN_models/a.h5')
Ptest = []
for i in model.predict(x=Xtest):
		Ptest.append(np.argmax(i))
Ttest = []
for i in Ytest:
		Ttest.append(np.argmax(i))
f1_score = f1_score(Ttest, Ptest, average='macro')
print('F値：', f1_score)



score = model.evaluate(Xtest, Ytest, verbose=1)
# accracyのグラフ
plt.plot(history.history['acc'],label="training")
plt.plot(history.history['val_acc'],label="validation")
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='upper right')
plt.savefig("./acc.png")
plt.close()
# lossのグラフ
plt.plot(history.history['loss'],label="training")
plt.plot(history.history['val_loss'],label="validation")
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig('./loss.png')
plt.close()