# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:26:33 2019

@author: Sardor
"""

import tkinter as tk
from PIL import Image
import tkinter.filedialog as filedialog
from tkinter import messagebox
import cv2
import numpy as np
import glob
import os.path
from tkinter.ttk import Frame, Button, Label
from threading import Thread
from keras import backend as K

class Mitosis_GUI(Frame):           
    def __init__(self):
        super().__init__()   
        self.initUI()
        
        
    def initUI(self):
      
        self.master.title("Mitosis Detection")
        self.pack(fill=tk.BOTH, expand=True)
        
        # divide main window to 2 frames
        self.upper_frame = Frame(self)
        self.upper_frame.pack()
        self.bottom_frame = Frame(self)
        self.bottom_frame.pack( side = tk.BOTTOM, fill=tk.X )
        
        # grid configuration for frame 1
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=4)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(6, pad=7)
        
        # GUI variables
        self.folder_path = tk.StringVar()
        self.regex_str = tk.StringVar()
        self.chck_box_regex_val = tk.IntVar()
        self.chck_box_csv_val = tk.IntVar()
        self.csv_path_str = tk.StringVar()
        self.statusbar_var = tk.StringVar() 
        
        # model variables
        self.weights_filename = 'weights_multispectral_v2'
        self.di_vetor_filename = 'avg.npy'
        self.IMG_SIZE = 60
        self.RESIZE_SHAPE = (60, 60)
        self.IMAGE_SHAPE = 1360,1360
        
        # create group for input
        group = tk.LabelFrame(self.upper_frame, text="File selection", padx=5, pady=5)
        group.grid(padx=10, pady=10)
        
        lbl_browse = Label(group, text="Browse 10-band images folder:")
        lbl_browse.grid(sticky=tk.W, pady=4, padx=5)
        entry_dir = tk.Entry(group, width=30, textvariable=self.folder_path)
        entry_dir.grid(row=0, column=1, sticky=tk.W, pady=4, padx=5)
        dir_browse_btn = Button(group, text="Browse",command=self.browse_dir_button)
        dir_browse_btn.grid(row=0, column=2, sticky=tk.W)
        
        chck_box_rgx = tk.Checkbutton(group, text="Use Regex",
                                      variable=self.chck_box_regex_val,
                                      command=self.check_btn_regex)
        chck_box_rgx.grid(row=1, sticky=tk.W)
        
        lbl_regex = Label(group, text="Images Regex:")
        lbl_regex.grid(row=2,sticky=tk.W, pady=4, padx=5)
        self.entry_regex = tk.Entry(group, width=30, textvariable=self.regex_str,
                               state='disabled')
        self.entry_regex.grid(row=2, column=1, sticky=tk.W,pady=4, padx=5)
        lbl_regex_example = Label(group, text="Regex Example: M00_01a_0[0-9]06")
        lbl_regex_example.grid(row=3,column=1,sticky=tk.W, pady=4, padx=5)
        
        chck_box_csv = tk.Checkbutton(group, text="Use Annotation CSV",
                                      variable=self.chck_box_csv_val,
                                      command=self.check_btn_csv)
        chck_box_csv.grid(row=4, sticky=tk.W)
        
        lbl_csv = Label(group, text="Annotation path:")
        lbl_csv.grid(row=5,sticky=tk.W, pady=4, padx=5)
        entry_csv = tk.Entry(group, width=30, textvariable=self.csv_path_str,
                               state='disabled')
        entry_csv.grid(row=5, column=1, sticky=tk.W,pady=4, padx=5)
        self.csv_browse_btn = Button(group,state='disabled', text="Browse CSV",command=self.browse_file_button)
        self.csv_browse_btn.grid(row=5, column=2,sticky=tk.W)
        

        self.filter_size_scale = tk.Scale(group, from_=53, to=60, orient=tk.HORIZONTAL)
        self.filter_size_scale.grid(row=6, column=0, pady=4, padx=5, sticky=tk.W)    
        self.area_expander_scale = tk.Scale(group, from_=13, to=18, orient=tk.HORIZONTAL)
        self.area_expander_scale.set(16)
        self.area_expander_scale.grid(row=6, column=1, pady=4, padx=5, sticky=tk.W)
        
        lbl_filter_size = Label(group, text="Filter Divider Value")
        lbl_filter_size.grid(row=7, column=0, sticky=tk.W, pady=4, padx=5)
        lbl_area_expander = Label(group, text="Candidate Area Expansion")
        lbl_area_expander.grid(row=7, column=1, sticky=tk.W, pady=4, padx=5)

        
        
        self.run_btn = Button(self.upper_frame, text="Run Model",command=self.run_btn_func)
        self.run_btn.grid(row=8, column=0, sticky=tk.N)
        
        self.progressbar = tk.ttk.Progressbar(self.bottom_frame, mode='indeterminate')
        self.progressbar.pack(fill=tk.X)
        self.status_bar=tk.Label(self.bottom_frame, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                           textvariable=self.statusbar_var)
        self.statusbar_var.set('Ready...')
        self.status_bar.pack(side = tk.BOTTOM,fill=tk.X)        
    
    
    def browse_dir_button(self):
        filename = filedialog.askdirectory()
        self.folder_path.set(filename)
        print(filename)
        
    def browse_file_button(self):
        filename = filedialog.askopenfilename(title = "Select CSV file",
                                              filetypes = (("CSV files","*.csv"),
                                                           ("all files","*.*")))
        self.csv_path_str.set(filename)
        print(self.csv_path_str.get())
    
    
    def check_btn_regex(self):
        if self.chck_box_regex_val.get() == 0:
            self.entry_regex.config(state='disabled')
        else:
            self.entry_regex.config(state='normal')
            
    def check_btn_csv(self):
        if self.chck_box_csv_val.get() == 0:
            self.csv_browse_btn.config(state='disabled')
        else:
            self.csv_browse_btn.config(state='normal')
    
    
    def update_statusbar(self, new_str):
        self.statusbar_var.set(new_str)
        self.status_bar.configure(text=self.statusbar_var)
    
    
    def run_btn_func(self):
        self.update_statusbar('Loading Files...')
        self.run_btn.config(state='disabled')
        self.progressbar.start()
        
        t_d1 = Thread(target=self.d1, args=())
        t_d1.start()
    
    
    def load_files(self):
        if self.chck_box_regex_val.get() == 0:
            files = glob.glob(self.folder_path.get()+'/*.bmp')
        elif self.regex_str.get() != '' :
            files = glob.glob(self.folder_path.get()+'/'+self.regex_str.get()+'.bmp')
        else:
            str_message = "Empty Regex"
            messagebox.showerror("Error", str_message)
            self.statusbar_var.set('Ready... | '+str_message)
            self.run_btn.config(state='normal')
            self.progressbar.stop()
            return 0, None
        
        if os.path.exists(self.di_vetor_filename)==False:
            str_message = "Discriminative vector file '"+self.di_vetor_filename+"' is missing"
            messagebox.showerror("Error", str_message)
            self.statusbar_var.set('Ready... | '+str_message)
            self.run_btn.config(state='normal')
            self.progressbar.stop()
            return 0, None
        
        if os.path.exists(self.weights_filename)==False:
            str_message = "Model weights '"+self.weights_filename+"' is missing"
            messagebox.showerror("Error", str_message)
            self.statusbar_var.set('Ready... | '+str_message)
            self.run_btn.config(state='normal')
            self.progressbar.stop()
            return 0, None
        
        if os.path.exists('mitos_model.py')==False:
            str_message = "CNN model file \'mitos_model.py\' is missing"
            messagebox.showerror("Error", str_message)
            self.statusbar_var.set('Ready... | '+str_message)
            self.run_btn.config(state='normal')
            self.progressbar.stop()
            return 0, None
        
        if self.chck_box_csv_val.get() == 1 and self.csv_path_str.get()=="":
            str_message = "Annotation file not selected"
            messagebox.showerror("Error", str_message)
            self.statusbar_var.set('Ready... | '+str_message)
            self.run_btn.config(state='normal')
            self.progressbar.stop()
            return 0, None
        
        if len(files)==10:
            return 1, files
        elif len(files)==0:
            str_message = "Could not find images"
            messagebox.showerror("Error", str_message)
            self.statusbar_var.set('Ready... | '+str_message)
            self.run_btn.config(state='normal')
            self.progressbar.stop()
            return 0, None
        else:
            str_message = "Images are not equal to 10 bands"
            messagebox.showerror("Error", str_message)
            self.statusbar_var.set('Ready... | '+str_message)
            self.run_btn.config(state='normal')
            self.progressbar.stop()
            return 0, None
        
    
    def d1(self):
        status, files_list = self.load_files()
        if status == 0:
            return
        
        self.update_statusbar('Generating DI...')
        t_di = Thread(target=self.generate_di, args=(files_list,))
        t_di.start()
        
    
    def generate_di(self,files_list):
        self.images = []
        for image_name in files_list:
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            self.images.append(image)
        
        if len(self.images)!=10:
            messagebox.showerror("Error", "Please check images directory.\
                                    There should be 10 bitmap images only")
            return
        
        discriminative_vector = np.load(self.di_vetor_filename)
        disc_image = None
        for coef,image in zip(discriminative_vector,self.images):
            if(disc_image is None):
                disc_image = (image*coef)
            else:
                disc_image += (image*coef)
        
        norm_disc_image = disc_image*255.0/disc_image.max()
        norm_disc_image = norm_disc_image.astype('uint8')
        self.di_image_gray = norm_disc_image


        self.update_statusbar('Model is predicting mitosis...')
        t_detect = Thread(target=self.detect_mitosis, args=())
        t_detect.start()
    
    
    def detect_mitosis(self):
        def expand_area(x,y,w,h,expand_pixels):
            if y-expand_pixels>=0:
                y -= expand_pixels
            if x-expand_pixels>=0:
                x -= expand_pixels
            if y+w+2*expand_pixels<=self.IMAGE_SHAPE[0]:
               w += 2*expand_pixels
            if x+h+2*expand_pixels<=self.IMAGE_SHAPE[1]:
               h += 2*expand_pixels
            return x,y,w,h
        
        try:
            from mitos_model import MITOSIS_CNN
            try:
                model = MITOSIS_CNN(self.IMG_SIZE, weights=self.weights_filename, channels=10)
                cvs_filename = self.csv_path_str.get()
                if self.chck_box_csv_val.get() == 1:
                    with open(cvs_filename,'r') as csv_file:
                        print('opened csv file',cvs_filename)
                        mitosis_pixels = []
                        for line in csv_file:
                            splitted_line = line.split(',')
                            pixel_data = []
                            for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                                pixel_data.append(( int(x), int(y) ))
                            mitosis_pixels.append(pixel_data)
                    image_mask = np.zeros(self.IMAGE_SHAPE,dtype='uint8')
                    for pixel_data in mitosis_pixels:
                        for pixels in pixel_data:
                            image_mask[pixels[1],pixels[0]] = 255
                    _, mask_contours, _ = cv2.findContours(image_mask,cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                else:
                    image_mask = None
                
                image_gray = self.di_image_gray
                image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
                
                #filtering
                kernel_size = (11,11)
                kernel = np.ones(kernel_size,np.float32)/self.filter_size_scale.get()
                filter2d = cv2.filter2D(image_gray,-1,kernel)

#                cv2.imwrite('filet2d.bmp',filter2d)
                
                #k-means
                kmeans_data = np.float32(filter2d.flatten())
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _,labels,centers = cv2.kmeans(kmeans_data,2,None,criteria,10,
                                              cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                kmeans = centers[labels.flatten()]
                kmeans = kmeans.reshape((image_gray.shape))
                
#                cv2.imwrite('kmeans.bmp',kmeans)
                
                #thresholding
                _, thresh = cv2.threshold(kmeans,240,255,cv2.THRESH_BINARY_INV)
#                thresh = cv2.bitwise_not(thresh)
                
#                cv2.imwrite('thresh.bmp',thresh)
                
                # find contours in the binary image
                _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
                candidates = []
                for contour in contours:
                    (x,y,w,h) = cv2.boundingRect(contour)
                    if w>10 and w<90 and h>10 and h<80:
                        candidates.append(contour)
                print('Candidates count:',len(candidates))
                
                for i,contour in enumerate(candidates):
                    (ox,oy,ow,oh) = cv2.boundingRect(contour)
                    x,y,w,h = expand_area(ox,oy,ow,oh,self.area_expander_scale.get())
                    candidate = np.empty((self.RESIZE_SHAPE[0],self.RESIZE_SHAPE[1],10),
                                         dtype='uint8')
                    for band,img in enumerate(self.images):
                        area = img[ y:y+h, x:x+w ].copy()
                        area = cv2.resize(area, self.RESIZE_SHAPE)
                        candidate[:,:,band] = area
                
                    res = model.predict_class(candidate)
                    res = [round(res[0], 5),round(res[1], 5)]
                    if res[0]>0.85:
                        print(res)
                        cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (255,0,0), 2)
                        if image_mask is not None:
                            image_mask_area = image_mask[ y:y+h, x:x+w ]
                            if(np.mean(image_mask_area) != 0):
                                image_mask[ y:y+h, x:x+w ] *= 0
                                cv2.circle(image_rgb,(int(x+w/2), int(y+h/2)), 60, (100,255,50), 4) 
                    elif res[0]>0.75:
                        cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (50,240,180), 2)
                    else:
                        cv2.rectangle(image_rgb, (x,y), (x+w,y+h), (255,255,30), 2)
                    
                if image_mask is not None:
                    _, mask_contours, _ = cv2.findContours(image_mask,cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                    for cont in mask_contours:
                        (x,y,w,h) = cv2.boundingRect(cont)
                        cv2.circle(image_rgb,(int(x+w/2), int(y+h/2)), 60, (100,0,50), 4)
                image_rgb_PIL = Image.fromarray(image_rgb)
                image_rgb_PIL.show()
                cv2.imwrite('image_rgb.bmp',image_rgb)
                self.statusbar_var.set('Ready...')
                self.run_btn.config(state='normal')
                self.progressbar.stop()
                K.clear_session()
                
            except Exception as e:
                messagebox.showerror("Error",e)
                self.statusbar_var.set('Ready... | '+str(e))
                self.run_btn.config(state='normal')
                self.progressbar.stop()
                print(e)
                return 0, None
        except:
            str_message = "MITOSIS_CNN class is missing in file \'mitos_model.py\'"
            messagebox.showerror("Error",str_message)
            self.statusbar_var.set('Ready... | '+str_message)
            self.run_btn.config(state='normal')
            self.progressbar.stop()
            return 0, None


def main():
#  M00_01a_0[0-9]06.bmp
    root = tk.Tk()
    root.iconbitmap('icon.ico')
    root.geometry("500x360+200+200")
    app = Mitosis_GUI()
    root.mainloop()
    K.clear_session()
    

if __name__ == '__main__':
    main()  




#    M00_01a_0[0-9]06
#
#
#root = tk.Tk()
#img = ImageTk.PhotoImage(di_image)
#group = tk.LabelFrame(root, text="Group", padx=5, pady=5)
#group.pack(padx=10, pady=10)
##group2 = tk.LabelFrame(root, text="Group2", padx=20, pady=5)
##group.pack(padx=10, pady=10)
#
#
#folder_path = tk.StringVar()
#e = tk.StringVar()
#
#w1 = tk.Label(root, image=img).pack(side="right",fill = "both", expand = "yes")
#
#tk.Label(group, text="Multispectral images regex:").pack(side='left', padx=5, pady=10)
#tk.Entry(group, width=40, textvariable=e).pack(side='left')
#
#tk.Label(group, text="Multispectral images directory:").pack(side='bottom', padx=5, pady=10)
#tk.Entry(group, width=40, textvariable=folder_path).pack()
##button = tk.Button(text="Browse", fg="blue",command=browse_button).pack(side='bottom')
#
#
#
#root.mainloop()