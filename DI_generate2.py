# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:56:52 2018

@author: Sardor
"""

import cv2
import glob
import numpy as np

PATH_WRITE_DI = 'DI_IMAGES/3_channel/'
PATH_MAIN = 'DATASET_RAW/'


discriminative_vector = np.load('avg.npy')

pairs = [[0,2,3,7,8,9],
         [1,4],
         [5,6]]

#pairs = [[2,3,7,8],
#         [1,4,5],
#         [0,9,6]]

for main_directory in glob.glob(PATH_MAIN+'*/'):
    main_directory_name = main_directory[12:-1]
    for part in 'abcd':       
        #iterate over 10 spectral band of image
        images_regex = main_directory + main_directory_name +'/' + main_directory_name \
                        + part + '_0[0-9]06.bmp'
        images = []
        for image_name in glob.glob(images_regex):
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            images.append(image)
            
        disc_image0 = np.zeros((1360,1360),dtype='float64')
        disc_image1 = np.zeros((1360,1360),dtype='float64')
        disc_image2 = np.zeros((1360,1360),dtype='float64')
        for ii,image in enumerate(images):
            if ii in pairs[0]:
                disc_image0 += (image*discriminative_vector[ii])
            elif ii in pairs[1]:
                disc_image1 += (image*discriminative_vector[ii])
            elif ii in pairs[2]:
                disc_image2 += (image*discriminative_vector[ii])
        
        norm_disc_image0 = disc_image0 * 255.0/disc_image0.max()
        norm_disc_image0 = norm_disc_image0.astype('uint8')
#        cv2.imwrite(PATH_WRITE_DI+image_name[-16:-9]+'0.bmp',norm_disc_image0)
        
        norm_disc_image1 = disc_image1 * 255.0/disc_image1.max()
        norm_disc_image1 = norm_disc_image1.astype('uint8')
#        cv2.imwrite(PATH_WRITE_DI+image_name[-16:-9]+'1.bmp',norm_disc_image1)
        
        norm_disc_image2 = disc_image2 * 255.0/disc_image2.max()
        norm_disc_image2 = norm_disc_image2.astype('uint8')
#        cv2.imwrite(PATH_WRITE_DI+image_name[-16:-9]+'2.bmp',norm_disc_image2)
        
        disc_image_rgb = np.zeros((1360,1360,3),dtype='uint8')
        disc_image_rgb[:,:,0] = norm_disc_image0
        disc_image_rgb[:,:,1] = norm_disc_image1
        disc_image_rgb[:,:,2] = norm_disc_image2
        cv2.imwrite(PATH_WRITE_DI+image_name[-16:-9]+'.bmp',disc_image_rgb)
        
#        break
#    break