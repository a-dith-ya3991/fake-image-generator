import cv2
import numpy as np
import os
import tensorflow as tf
def images_to_dataset(path,img_size=(64,64)):
    
    images_list=os.listdir(path)
    images=[]   
    for _,i in enumerate(images_list):
        if '.jpg' in i:
            img=cv2.imread(os.path.join(path,i))
            img=cv2.resize(img,img_size)
            images.append(img)
    
    images= (tf.cast(images, tf.float32) - 127.5) / 127.5
    return images