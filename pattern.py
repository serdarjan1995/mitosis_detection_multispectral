# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 00:28:36 2018

@author: Sardor
"""

import glob
import numpy as np
import cv2

PATH_MAIN = 'DATASET_RAW/'

images = []


for main_directory in glob.glob(PATH_MAIN+'*/'):
    main_directory_name = main_directory[12:-1]
    HPF_NUMBER = []
    if(main_directory_name != 'M01_07'):
        continue
    
    for hpf_number_csv in glob.glob(main_directory+'*.csv'):
        hpf_number_csv_name = hpf_number_csv[19:-4]
        #parse cv file:
        with open(hpf_number_csv,'r') as csv_file:
            print('opened csv file',hpf_number_csv)
            mitosis_pixels = []
            for line in csv_file:
                splitted_line = line.split(',')
                pixel_data = []
                for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                    pixel_data.append(( int(x), int(y) ))
                mitosis_pixels.append(pixel_data)
        
        # read images with mitosis 
        images_regex = main_directory+main_directory_name+'/' \
                        + hpf_number_csv_name[:-4] \
                        + '0[0-9]'+hpf_number_csv_name[-2:]+'.bmp'
        for image_name in glob.glob(images_regex):
            SPECTRAL_BAND = []
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            image_mask = image.copy()
            image_mask *= 0
            for pixel_data in mitosis_pixels:
                for pixels in pixel_data:
                    image_mask[pixels[1],pixels[0]] = 255
                    SPECTRAL_BAND.append(image[pixels[1],pixels[0]])
            
            HPF_NUMBER.append(SPECTRAL_BAND)
            
            resize_height = int(image.shape[0]/2)
            resize_width = int(image.shape[1]/2)
            image = cv2.resize( image, (resize_height, resize_width) )
            
            image_mask = cv2.resize(image_mask, (resize_height, resize_width))
            
            # for optimizing contours
            kernel = np.ones((3,3),np.uint8)
            image_mask = cv2.dilate(image_mask, kernel, iterations=1)
            image_mask = cv2.erode(image_mask, kernel, iterations=1)
            
            
            # find contours in the binary image
            _, contours, _ = cv2.findContours(image_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            # change colorspace to bgr
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
            print('contour len',len(contours))
            cv2.drawContours(image, contours, -1, (255,0,255), 2)
            for c in contours:
                # calculate moments for each contour
                M = cv2.moments(c)

                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(image,(x-3,y-3),(x+w+3,y+h+3),(100,255,150),3)
                
            images.append(image)
            img_concat = np.concatenate((image, image_mask), axis=1)
            cv2.imshow(image_name,img_concat)
            key_to_quit = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key_to_quit==27:    # Esc key to stop
                break
            
        
#        break
    
#    break
print('all done')
for i in range(0,len(images)):
    resize_height = int(images[i].shape[0]/2.5)
    resize_width = int(images[i].shape[1]/2.5)
    images[i] = cv2.resize( images[i], (resize_height, resize_width) )

img_concat = np.concatenate((images[0], images[1],images[2], images[3],images[4]), axis=1)
img_concat2 = np.concatenate((images[5], images[6],images[7], images[8],images[9]), axis=1)
img_concat3 = np.concatenate((img_concat,img_concat2), axis=0)

cv2.imshow(image_name,img_concat3)
key_to_quit = cv2.waitKey(0)
cv2.destroyAllWindows()