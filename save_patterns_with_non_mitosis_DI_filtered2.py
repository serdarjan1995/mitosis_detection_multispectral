# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 00:28:36 2018

@author: Sardor
"""

import glob
import numpy as np
import cv2
import scipy.io
from random import shuffle

PATH_DI = 'DI_IMAGES/3_channel/'
PATH_DI_1ch = 'DI_IMAGES/'
PATH_MAIN = 'DATASET_RAW/'
PATH_PATTERNS = 'NONMITOS_PATTERNS_DI_3channel/'

IMAGE_SHAPE = (1360,1360)
RESIZE_SHAPE = (60, 60)

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

#def expand_area(x,y,w,h,expand_val):
#    print(x,y,w,h)
#    new_x,new_y,new_w,new_h = x-15,y-15,RESIZE_SHAPE[0],RESIZE_SHAPE[1]
#    if new_y<0:
#        new_y = y
#    if new_x<0:
#        new_x = x
#    if y+new_w>IMAGE_SHAPE[0]:
#        print('y+new_w',y+new_w)
#        new_w = IMAGE_SHAPE[0]-y
#        print('new_w',new_w)
#    if x+new_h>IMAGE_SHAPE[1]:
#        print('x+new_h',x+new_h)
#        new_h = IMAGE_SHAPE[1]-x
#        print('new_h',new_h)
#    return new_x,new_y,new_w,new_h

images_list = glob.glob(PATH_MAIN+'*/')
#shuffle(images_list)
for main_directory in images_list:
    main_directory_name = main_directory[12:-1]
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
        
        images_filename = PATH_DI_1ch + main_directory_name + part + '.bmp'
        image_gray = cv2.imread(images_filename,0)
        print('opened image',images_filename)
        images_filename = PATH_DI + main_directory_name + part + '.bmp'
        image_rgb = cv2.imread(images_filename)
        
        #filtering
        kernel_size = (11,11)
        kernel = np.ones(kernel_size,np.float32)/55
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
            if w>10 and w<90 and h>10 and h<70:
                candidates.append(contour)
                
#        for contour in candidates:
#            (x,y,w,h) = cv2.boundingRect(contour)
#            x,y,w,h = expand_area(x,y,w,h,10)
#            cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (51,200,255), 2)
#        
#        resize_val = int(image_gray.shape[0]/2)
#        filter2d = cv2.resize( filter2d, (resize_val, resize_val) )
#        kmeans = cv2.resize( kmeans, (resize_val, resize_val) )
#        image_gray = cv2.resize( image_gray, (resize_val, resize_val) )
#        image_rgb = cv2.resize( image_rgb, (resize_val, resize_val) )
##        gray = cv2.cvtColor(kmeans,cv2.COLOR_BGR2GRAY)
#        
#        cv2.imshow(images_filename+' filer2d',filter2d)
#        cv2.imshow(images_filename+' kmeans',kmeans)
##        cv2.imshow(images_filename+' gray',image_gray)
#        cv2.imshow(images_filename+' rgb',image_rgb)
#        key_to_quit = cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        if key_to_quit==27:    # Esc key to stop
#            break
#    break
        
        if len(candidates)>30:
            shuffle(candidates)
            candidates = candidates[:30]
                
        for i,contour in enumerate(candidates):
            write_to_file = False
            (x,y,w,h) = cv2.boundingRect(contour)
            x,y,w,h = expand_area(x,y,w,h,10)
            if image_mask is None:
                non_mitosis_area = image_rgb[y:y+w, x:x+h].copy()
                non_mitosis_area = cv2.resize(non_mitosis_area, RESIZE_SHAPE)
                write_to_file = True
            else:
                image_mask_area = image_mask[y:y+w, x:x+h]
                if(np.mean(image_mask_area) == 0):
                    write_to_file = True
                    non_mitosis_area = image_rgb[y:y+w, x:x+h].copy()
                    non_mitosis_area = cv2.resize(non_mitosis_area, RESIZE_SHAPE)

            if write_to_file == True:
                image_save_filename = PATH_PATTERNS + main_directory_name + part+'_' + str(i)
                cv2.imwrite(image_save_filename + '.bmp',non_mitosis_area)
#                np.save(image_save_filename, non_mitosis_area)
#                scipy.io.savemat(image_save_filename, {'nonmitosis_area':non_mitosis_area})
                    
#        break
#    break




#            for (i,mitos) in enumerate(mitosis):
#                cv2.imshow(image_name,mitos)
#                key_to_quit = cv2.waitKey(0)
#                cv2.destroyAllWindows()
#                if key_to_quit==27:    # Esc key to stop
#                    break
        
#        break

