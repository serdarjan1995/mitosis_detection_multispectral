# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:19:10 2018

@author: Sardor
"""

import cv2
import numpy as np
import glob
from random import shuffle
from mitos_model_v2 import MITOSIS_CNN


IMG_SIZE = 60
RESIZE_SHAPE = (60, 60)
IMAGE_SHAPE = 1360,1360
weights_path = 'weights_multispectral_v2'
images_di_path = 'DI_IMAGES/'
PATH_MAIN = 'DATASET_RAW/'
MITOSIS_PATH = 'MITOS_M/'
NONMITOSIS_PATH = 'NONMITOS_M/'

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

model = MITOSIS_CNN(IMG_SIZE, weights=weights_path, channels=10)
#model.generate_train_data(MITOSIS_PATH,NONMITOSIS_PATH)
#model.train_model(epochs=30,min_delta=0.01,cross_validation=0.2)


dirs = glob.glob(PATH_MAIN+'*/')
#shuffle(dirs)
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for main_directory in dirs:
    main_directory_name = main_directory[12:-1]
    key_to_quit = 0
    for part in 'abcd':
        csv_name = main_directory + main_directory_name + part + '_0607.csv'
        image_mask = None
        #check whether there is csv annotation, if is then generate mitosis mask
        if csv_name in glob.glob(main_directory+'*'):
            with open(csv_name,'r') as csv_file:
                print('opened csv file',csv_name)
                mitosis_pixels = []
                for line in csv_file:
                    splitted_line = line.split(',')
                    pixel_data = []
                    for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                        pixel_data.append(( int(x), int(y) ))
                    mitosis_pixels.append(pixel_data)
            
            image_mask = np.zeros(IMAGE_SHAPE,dtype='uint8')
            for pixel_data in mitosis_pixels:
                for pixels in pixel_data:
                    image_mask[pixels[1],pixels[0]] = 255
            _, mask_contours, _ = cv2.findContours(image_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    
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
        kernel = np.ones(kernel_size,np.float32)/56
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
        print('Candidates count:',len(candidates))
        
        for i,contour in enumerate(candidates):
            (ox,oy,ow,oh) = cv2.boundingRect(contour)
            x,y,w,h = expand_area(ox,oy,ow,oh,15)
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
#            candidate = np.reshape(candidate,(1,
#                                              candidate.shape[0],
#                                              candidate.shape[1],
#                                              candidate.shape[2]))
#            candidate = candidate.astype('float32')/255
            res = model.predict_class(candidate)
            res = [round(res[0], 5),round(res[1], 5)]
            if res[0]>0.8:
                cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (0,0,255), 2)
#                print('\t',res)
                if image_mask is not None:
                    image_mask_area = image_mask[ y:y+h, x:x+w ]
                    if(np.mean(image_mask_area) != 0):
                        true_positive += 1
                        image_mask[ y:y+h, x:x+w ] *= 0
#                        for mm in mask_contours:
#                            (mx,my,mw,mh) = cv2.boundingRect(mm)
#                            if mx+mw>=ox+ow and my+mh>=oy+oh:
#                                image_mask[ my:my+mh, mx:mx+mw ] *= 0
#                                print(np.mean(image_mask[ my:my+mh, mx:mx+mw ]))
                        cv2.circle(image_rgb,(int(x+w/2), int(y+h/2)), 60, (50,0,100), 4) 
                    else:
                        false_positive += 1
                else:
                    false_positive += 1
            elif res[0]>0.7:
                cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (180,240,50), 2)
                print('\t',res)
            else:
                cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (51,255,255), 1)
                print(res)
                true_negative += 1
        
        if image_mask is not None:
            _, mask_contours, _ = cv2.findContours(image_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            false_negative += len(mask_contours)
            for cont in mask_contours:
                (x,y,w,h) = cv2.boundingRect(cont)
                cv2.circle(image_rgb,(int(x+w/2), int(y+h/2)), 60, (100,0,50), 4)
#            cv2.imshow('threshm '+images_filename[-11:],image_mask)
        print('TP',true_positive)
        print('TN',true_negative)
        print('FP',false_positive)
        print('FN',false_negative)
        
#        resize_val = int(image_gray.shape[0]/2)
#        kmeans = cv2.resize( kmeans, (resize_val, resize_val) )
#        thresh = cv2.resize( thresh, (resize_val, resize_val) )
#        image_gray = cv2.resize( image_gray, (resize_val, resize_val) )
#        image_rgb = cv2.resize( image_rgb, (resize_val, resize_val) )
#        
#        while True:
#            cv2.imshow('thresh '+images_filename[-11:],thresh)
##            cv2.imshow('kmeans '+images_filename[-11:],kmeans)
#            cv2.imshow('image_rgb '+images_filename[-11:],image_rgb)
#        
#            key_to_quit = cv2.waitKey(0)
#            cv2.destroyAllWindows()
#            if key_to_quit==ord("q") or key_to_quit==27:    # Space or q key to stop
#                break
#        if key_to_quit==ord("q"):    # Esc key to stop image chain
#            break
#    if key_to_quit==ord("q"):    # Esc key to stop image chain
#        break