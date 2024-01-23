import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2DTranspose,Dropout,BatchNormalization,Conv2D,LeakyReLU,Reshape,Input,SeparableConv2D,Flatten
from  tensorflow.keras.models import Model

def generator():
    input_layer=Input(shape=(100))
    dense=Dense(64*64*3,activation=LeakyReLU(0.001))(input_layer)
    reshape=Reshape((64,64,3))(dense)

    conv=Conv2DTranspose(128,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.0001))(reshape)
    conv=Conv2DTranspose(128,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.001))(conv)
    drop=Dropout(0.0001)(conv)
    conv=Conv2DTranspose(512,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.001))(drop)
    conv=Conv2DTranspose(512,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.0001))(conv)
    drop=Dropout(0.001)(conv)
    norm=BatchNormalization()(drop)
    conv=Conv2D(512,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.001))(norm)
    conv=Conv2DTranspose(512,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.0002))(conv)
    drop=Dropout(0.0001)(conv)
    norm=BatchNormalization()(drop)
    conv=Conv2DTranspose(512,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.001))(norm)
    conv=Conv2DTranspose(512,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.0002))(conv)
    drop=Dropout(0.0001)(conv)
    norm=BatchNormalization()(drop)
    drop=Dropout(0.0001)(conv)
    norm=BatchNormalization()(drop)
    conv=Conv2DTranspose(256,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.001))(norm)
    conv=Conv2DTranspose(256,3,1,padding='same',use_bias=False,activation=LeakyReLU(0.001))(conv)

    drop=Dropout(0.0001)(conv)
    norm=BatchNormalization()(drop)
    output_layer=Conv2DTranspose(3,3,1,padding='same',use_bias=False,activation='tanh')(norm)
    model=Model(input_layer,output_layer)
    return model
    
def discriminator():
    input_layer=Input(shape=(64,64,3))
    conv=Conv2D(128,3,2 ,use_bias=False,activation=LeakyReLU(0.0001))(input_layer)
    conv=Conv2D(128,3,2 ,use_bias=False,activation=LeakyReLU(0.001))(conv)
    drop=Dropout(0.0001)(conv)
    conv=SeparableConv2D(512,3,1 ,use_bias=False,activation=LeakyReLU(0.001))(drop)
    conv=SeparableConv2D(512,3,1 ,use_bias=False,activation=LeakyReLU(0.0001))(conv)
    drop=Dropout(0.001)(conv)
    norm=BatchNormalization()(drop)
    conv=Conv2D(512,3,1 ,use_bias=False,activation=LeakyReLU(0.001))(norm)
    conv=SeparableConv2D(512,3,1 ,use_bias=False,activation=LeakyReLU(0.0002))(conv)
    drop=Dropout(0.0001)(conv)
    norm=BatchNormalization()(drop)
    conv=Conv2D(512,3,1 ,use_bias=False,activation=LeakyReLU(0.001))(norm)
    conv=SeparableConv2D(512,3,1 ,use_bias=False,activation=LeakyReLU(0.0002))(conv)
    drop=Dropout(0.0001)(conv)
    norm=BatchNormalization()(drop)
    
    conv=Conv2D(256,2,1 ,use_bias=False,activation=LeakyReLU(0.001))(norm)
    
    flatten=Flatten()(conv)
    
    dense=Dense(512,activation='relu')(flatten)

    output_layer=Dense(1,activation='sigmoid')(dense)
    disc_model=Model(input_layer,output_layer)
    return disc_model