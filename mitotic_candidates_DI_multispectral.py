# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:19:10 2018

@author: Sardor
"""

import cv2
import numpy as np
import glob
from random import shuffle
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt



IMG_SIZE = 60
RESIZE_SHAPE = (60, 60)
IMAGE_SHAPE = 1360,1360
weights_path = 'weights_multispectral_zoomed_dataset'
images_di_path = 'DI_IMAGES/'
PATH_MAIN = 'DATASET_RAW/'

def expand_area(x,y,w,h,expand_pixels):
    if y-expand_pixels>=0:
        y -= expand_pixels
    if x-expand_pixels>=0:
        x -= expand_pixels
    if y+w+2*expand_pixels<=IMAGE_SHAPE[0]:
       w += 2*expand_pixels
    if x+h+2*expand_pixels<=IMAGE_SHAPE[1]:
       h += 2*expand_pixels
    return x,y,w,h

inputShape = (IMG_SIZE, IMG_SIZE,10)

model = Sequential()
model.add(Convolution2D(96, kernel_size = (7, 7), activation='relu', input_shape=inputShape))
model.add(Convolution2D(384, kernel_size=(5,5), activation='relu'))
model.add(Convolution2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(256, kernel_size=(3,3), activation='relu'))
model.add(Convolution2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation = 'softmax'))
model.load_weights(weights_path)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

dirs = glob.glob(PATH_MAIN+'*/')
#shuffle(dirs)
for main_directory in dirs:
    main_directory_name = main_directory[12:-1]
    key_to_quit = 0
    for part in 'abcd':      
        #iterate over 10 spectral band of image
        images_regex = main_directory + main_directory_name +'/' + main_directory_name \
                        + part + '_0[0-9]06.bmp'
        images = []
        for image_name in glob.glob(images_regex):
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            images.append(image)
            
        images_filename = images_di_path + main_directory_name + part + '.bmp'
        image_gray = cv2.imread(images_filename,0)
        image_rgb = cv2.imread(images_filename)
        print('opened image',images_filename)
        
        #filtering
        kernel_size = (11,11)
        kernel = np.ones(kernel_size,np.float32)/52
        filter2d = cv2.filter2D(image_gray,-1,kernel)
        
        #k-means
        kmeans_data = np.float32(filter2d.flatten())
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _,labels,centers = cv2.kmeans(kmeans_data,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        kmeans = centers[labels.flatten()]
        kmeans = kmeans.reshape((image_gray.shape))
        
        
        #thresholding
        _, thresh = cv2.threshold(kmeans,240,255,cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)
        
        
        # find contours in the binary image
        _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            if w>12 and w<90 and h>12 and h<80:
                candidates.append(contour)
        print('candidates',len(candidates))
        for i,contour in enumerate(candidates):
            write_to_file = False
            (x,y,w,h) = cv2.boundingRect(contour)
            x,y,w,h = expand_area(x,y,w,h,15)
            candidate = np.empty((RESIZE_SHAPE[0],RESIZE_SHAPE[1],10), dtype='uint8')
            for band,img in enumerate(images):
                area = img[ y:y+h, x:x+w ].copy()
                area = cv2.resize(area, RESIZE_SHAPE)
                candidate[:,:,band] = area
                
#            img_concat = np.concatenate((candidate[:,:,0], candidate[:,:,1]),axis=1)
#            for i in range(2,10):
#                img_concat = np.concatenate((img_concat, candidate[:,:,i]),axis=1)
#            
#            while True and key_to_quit!=ord("q"):
#                cv2.imshow('test',img_concat)
#                key_to_quit = cv2.waitKey(0)
#                cv2.destroyAllWindows()
#                if key_to_quit==27 or key_to_quit==ord("q"):    # Esc key to stop image chain
#                    break
#            test = candidate.copy()
            candidate = np.reshape(candidate,[1,
                                              candidate.shape[0],
                                              candidate.shape[1],
                                              candidate.shape[2]])
            candidate = candidate.astype('float32')/255
            res = model.predict(candidate)[0]
            res = [round(res[0], 5),round(res[1], 5)]
            if res[0]>0.8:
                cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (0,0,255), 2)
                print('\t',res)
            elif res[0]>0.6:
                cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (180,240,50), 2)
                print('\t',res)
            else:
                cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (51,255,255), 1)
                print(res)

        resize_val = int(image_gray.shape[0]/2)
        kmeans = cv2.resize( kmeans, (resize_val, resize_val) )
        thresh = cv2.resize( thresh, (resize_val, resize_val) )
        image_gray = cv2.resize( image_gray, (resize_val, resize_val) )
        image_rgb = cv2.resize( image_rgb, (resize_val, resize_val) )
        
        while True:
            cv2.imshow('thresh '+images_filename[-11:],thresh)
            cv2.imshow('kmeans '+images_filename[-11:],kmeans)
            cv2.imshow('image_rgb '+images_filename[-11:],image_rgb)
        
            key_to_quit = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key_to_quit==ord("q") or key_to_quit==27:    # Space or q key to stop
                break
        if key_to_quit==ord("q"):    # Esc key to stop image chain
            break
    if key_to_quit==ord("q"):    # Esc key to stop image chain
        break