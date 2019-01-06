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

PATH_DI_1ch = 'DI_IMAGES/'
PATH_MAIN = 'DATASET_RAW/'
PATH_PATTERNS_NMC = 'NONMITOS_M/'
PATH_PATTERNS_MC = 'MITOS_M/'

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





for main_directory in glob.glob(PATH_MAIN+'*/'):
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
        
        #iterate over 10 spectral band of image
        images_regex = main_directory + main_directory_name +'/' + main_directory_name \
                        + part + '_0[0-9]06.bmp'
        images = []
        for image_name in glob.glob(images_regex):
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            images.append(image)
            
        images_filename = PATH_DI_1ch + main_directory_name + part + '.bmp'
        image_gray = cv2.imread(images_filename,0)
        print('opened image',images_filename)
        
        #filtering
        kernel_size = (11,11)
        kernel = np.ones(kernel_size,np.float32)/53
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

        if len(candidates)>120:
            shuffle(candidates)
            candidates = candidates[:120]
        
        
        for i,contour in enumerate(candidates):
            px,py,pw,ph = cv2.boundingRect(contour)
            expand_diffs = [12,13,14,15,16,17,18]
            is_mc = False
            if image_mask is not None:
                image_mask_area = image_mask[ py:py+ph, px:px+pw ]
                if(np.mean(image_mask_area) != 0):
                    is_mc = True
            
            for expand_diff in expand_diffs:
                x,y,w,h = expand_area(px,py,pw,ph,expand_diff)
                cell_pattern = np.empty((RESIZE_SHAPE[0],RESIZE_SHAPE[1],10),dtype='uint8')
                for band,img in enumerate(images):
                    candidate_area = img[ y:y+h, x:x+w ].copy()
                    candidate_area = cv2.resize(candidate_area, RESIZE_SHAPE)
                    cell_pattern[:,:,band] = candidate_area
                
                if is_mc == True:
                    image_save_filename = PATH_PATTERNS_MC + main_directory_name  \
                                           + part+'_can'+'_' + str(i)
                else:
                    image_save_filename = PATH_PATTERNS_NMC + main_directory_name \
                                           + part+'_' + str(i)
#                cv2.imwrite(image_save_filename + '.bmp',non_mitosis_area)
                np.save(image_save_filename+'_'+str(expand_diff), cell_pattern)
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

