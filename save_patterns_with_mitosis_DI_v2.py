# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 00:28:36 2018

@author: Sardor
"""

import glob
import numpy as np
import cv2
import scipy.io

PATH_DI = 'DI_IMAGES/'
PATH_MAIN = 'DATASET_RAW/'
PATH_PATTERNS = 'MITOS_PATTERNS_DI_v2/'

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

IMAGE_SHAPE = (1360,1360)
MITOSIS_AREA_SHAPE = (80,80)

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
                mitosis_count = 0
                for line in csv_file:
                    mitosis_count += 1
                    splitted_line = line.split(',')
                    pixel_data = []
                    for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                        pixel_data.append(( int(x), int(y) ))
                    mitosis_pixels.append(pixel_data)
            
            images_filename = PATH_DI + main_directory_name + part + '.bmp'
            image = cv2.imread(images_filename,0)
            print('opened image',images_filename)
            
            image_mask = np.zeros(IMAGE_SHAPE,dtype='uint8')
            for pixel_data in mitosis_pixels:
                for pixels in pixel_data:
                    image_mask[pixels[1],pixels[0]] = 255
           
            # for optimizing contours
            kernel = np.ones((3,3),np.uint8)
            image_mask = cv2.dilate(image_mask, kernel, iterations=2)
            image_mask = cv2.erode(image_mask, kernel, iterations=2)
            
            # find contours in the binary image
            _, contours, _ = cv2.findContours(image_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            boundingBoxes = [cv2.boundingRect(c) for c in contours]
                
            #check wheter contours count equals to mitosis count
            if(len(boundingBoxes) > mitosis_count):
                bbox_coors = []
                removing_bbox = []
                for boundingBox in boundingBoxes:
                    x,y,w,h = boundingBox
                    intersects = 0
                    for bbox in bbox_coors:
                        if bbox[0]<=x and (bbox[0]+bbox[2])>=(x+w) and \
                           bbox[1]<=y and (bbox[1]+bbox[3])>=(y+h) :
                                intersects = 1
                                removing_bbox.append(boundingBox)
                    if intersects == 0:
                        bbox_coors.append(boundingBoxes[0])
                #remove redundant bbox
                for rem in removing_bbox:
                    boundingBoxes.remove(rem)
                                
            #iterate over bboxes, save bbox bounded areas
            for (i,bbox) in enumerate(boundingBoxes):            
                x,y,w,h = bbox
                x,y,w,h = expand_area(x,y,w,h,5)
                mitos_area = image[ y:y+h, x:x+w ].copy()
                resized_data = cv2.resize(mitos_area, MITOSIS_AREA_SHAPE) 
                
                image_save_filename = PATH_PATTERNS + main_directory_name + part+'_' + str(i)
                cv2.imwrite(image_save_filename + '.bmp',resized_data)
                np.save(image_save_filename, resized_data)
#                scipy.io.savemat(image_save_filename, {'mitosis_area':resized_data})


#                cv2.imshow(image_name,mitos)
#                key_to_quit = cv2.waitKey(0)
#                cv2.destroyAllWindows()
#                if key_to_quit==27:    # Esc key to stop
#                    break


