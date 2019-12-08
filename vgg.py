from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D,MaxPooling2D,Dropout
from keras.models import Model


input_img = Input(shape=(224, 224, 3))
x=Conv2D(32,(3,3), strides=1,padding="same",activation="relu",input_shape=input(img)
