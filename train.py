from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam,SGD,Nadam
from keras.utils import np_utils


def cnn_model():
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


if __name__ == "__main__":
    