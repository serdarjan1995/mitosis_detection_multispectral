# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:56:52 2018

@author: Sardor
"""

import cv2
import glob
import numpy as np

PATH_WRITE_DI = 'DI_IMAGES/'
PATH_MAIN = 'DATASET_RAW/'


discriminative_vector = np.load('avg.npy')



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
        
        disc_image = None
        for coef,image in zip(discriminative_vector,images):
            if(disc_image is None):
                disc_image = (image*coef)
            else:
                disc_image += (image*coef)
        
        norm_disc_image = disc_image * 255.0/disc_image.max()
        norm_disc_image = norm_disc_image.astype('uint8')
        cv2.imwrite(PATH_WRITE_DI+image_name[-16:-9]+'.bmp',norm_disc_image)