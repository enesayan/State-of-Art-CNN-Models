from keras.layers import Input, Dense, Flatten,Activation,add
from keras.layers import Conv2D, MaxPooling2D, Dropout,BatchNormalization
from keras.models import Model

stride=1
axis=3
num_classes=2
def residual_layer(x,filters,pooling=False,dropout=0.0):
    temp = x
    temp = Conv2D(filters, (3, 3), strides=stride, padding="same")(temp)
    temp = BatchNormalization(axis=axis)(temp)
    temp = Activation("relu")(temp)
    temp = Conv2D(filters, (3, 3), strides=stride, padding="same")(temp)

    x = add([temp, Conv2D(filters, (3, 3), strides=stride, padding="same")(x)])
    if pooling:
        x = MaxPooling2D((2, 2))(x)
    if dropout != 0.0:
        x = Dropout(dropout)(x)
    x = BatchNormalization(axis=axis)(x)
    x = Activation("relu")(x)
    return x

input_img = Input(shape = (224,224,3))
x = input_img
x = Conv2D(16,(3,3),strides = stride,padding = "same")(x)
x = BatchNormalization(axis = axis)(x)
x = Activation("relu")(x)
x = residual_layer(x,32,dropout = 0.1)
x = residual_layer(x,32,dropout = 0.2)
x = residual_layer(x,32,dropout = 0.3,pooling = True)
x = residual_layer(x,64,dropout = 0.4)
x = residual_layer(x,64,dropout = 0.1,pooling = True)
x = residual_layer(x,256,dropout = 0.4)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024,activation = "relu")(x)
x = Dropout(0.4)(x)
x = Dense(num_classes,activation = "softmax")(x)
resnet_model = Model(input_img,x,name = "Resnet")

resnet_model.summary()
