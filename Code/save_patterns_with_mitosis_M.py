# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 00:28:36 2018

@author: Sardor
"""

import glob
import numpy as np
import cv2
import scipy.io

PATH_PATTERNS = 'MITOS_M/'
PATH_MAIN = 'DATASET_RAW/'

IMG_SIZE = 60
RESIZE_SHAPE = (60, 60)
IMAGE_SHAPE = 1360,1360


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
    
    for hpf_number_csv in glob.glob(main_directory+'*.csv'):
        hpf_number_csv_name = hpf_number_csv[19:-4]
        
        #parse csv file, read pixel coors
        with open(hpf_number_csv,'r') as csv_file:
            print('opened csv file',hpf_number_csv)
            mitosis_pixels = []
            mitosis_count = 0
            for line in csv_file:
                mitosis_count += 1
                splitted_line = line.split(',')
                pixel_data = []
                for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                    pixel_data.append(( int(x), int(y) ))
                mitosis_pixels.append(pixel_data)
        
        # read images with mitosis 
        images_regex = main_directory+main_directory_name+'/' \
                        + hpf_number_csv_name[:-4] \
                        + '0[0-9]06.bmp'
        image_mask = None
        images = []
        contours = []
        for image_name in glob.glob(images_regex):
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            
            # create mask
            if( image_mask is None):
                image_mask = image.copy()
                image_mask *= 0
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
            images.append(image)
            
        #iterate over bboxes, save bbox bounded areas
        for (num,bbox) in enumerate(boundingBoxes):            
            px,py,pw,ph = bbox
            expand_diffs = [8,9,10,11,12,13,14,15,16,17,18,19,20,21]
            for expand_diff in expand_diffs:
                x,y,w,h = expand_area(px,py,pw,ph,expand_diff)
                mitosis = []
                for im in images:
                    mitos_area = im[ y:y+h, x:x+w ].copy()
                    resized_data = cv2.resize(mitos_area, RESIZE_SHAPE) 
                    mitosis.append(resized_data)
                
                #create numpy matrix to save 10 spectral areas
                mitosis_pattern = np.empty((mitosis[0].shape[0],mitosis[0].shape[1],10),dtype='uint8')
                for (i,mitos) in enumerate(mitosis):
                    mitosis_pattern[:,:,i] = mitos
    #                cv2.imwrite(PATH_PATTERNS+hpf_number_csv_name[:-4]+str(num)+'_'+str(i)+'.bmp',mitos)
                horizontal_pattern = cv2.flip( mitosis_pattern, 0 )
                vertical_pattern = cv2.flip( mitosis_pattern, 1 )
                h_v_pattern = cv2.flip( mitosis_pattern, -1 )
                image_save_filename = PATH_PATTERNS+hpf_number_csv_name[:-4]+str(num)
                np.save(image_save_filename+'_'+str(expand_diff)+'_orig', mitosis_pattern)
                np.save(image_save_filename+'_'+str(expand_diff)+'_hflip', horizontal_pattern)
                np.save(image_save_filename+'_'+str(expand_diff)+'_vflip', vertical_pattern)
                np.save(image_save_filename+'_'+str(expand_diff)+'_hvflip', h_v_pattern)
#
#               scipy.io.savemat(image_save_filename, {'mitosis_area':mitosis_pattern_multispectral})

#            for (i,mitos) in enumerate(mitosis):
#                cv2.imshow(image_name,mitos)
#                key_to_quit = cv2.waitKey(0)
#                cv2.destroyAllWindows()
#                if key_to_quit==27:    # Esc key to stop
#                    break

#    break


