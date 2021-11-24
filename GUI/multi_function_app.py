#Import all libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join, isfile, exists
import progressbar
import numpy as np
import imageio
from PIL import Image
import argparse
import openslide as ops
import shapely.geometry as sg
#import pandas as pd
import cv2
import json
from math import sqrt
import time
import csv
from multiprocessing.dummy import Pool as ThreadPool

import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane

######################################################

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
from tkinter.simpledialog import askstring, askinteger
from tkinter import messagebox
import sys
from PIL import Image, ImageTk
from os.path import join
import imageio
from pathlib import Path

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def opening_page():
    global image1, gui
    gui = Tk()
    gui.geometry("530x400")
    gui.resizable(width=0, height=0)
    gui.title("Kather.ai application")
    photo1 = PhotoImage(file=resource_path('images\\display.GIF'))
    image1 = photo1.zoom(1, 1) 
    photo2 = PhotoImage(file=resource_path('images\\causion.GIF'))
    Label(gui, image=image1, bg="white", fg="white", ).grid(row=0, column=0, columnspan=4)
    gui.configure(background="black")
    image2 = photo2.subsample(6, 6)
    Label(gui, text="       This application is for research use and not for medical diagnosis", bg="black", fg="white", font="none 12 bold").grid(row=1, column=0,columnspan=4)
    Label(gui, image=image2, bg="white", fg="white", ).place(x=0, y=240)
    Button1=Button(gui, text="Tessellation",width=12,height=2,font="none 16 bold",justify="left",bg='#EBF5FB', command=tessellation).grid(row=2,column=0)
    Button1=Button(gui, text="Normalization",width=12,height=2,font="none 16 bold",justify="left",bg='#D6EAF8', command=normalization).grid(row=2,column=1)
    Button1=Button(gui, text="Thumb extraction",width=14,height=2,font="none 16 bold",justify="left",bg='#AED6F1',command=training).grid(row=2,column=2)
    Button2=Button(gui, text="Quit",width=6,height=1,font="none 16 bold",command=gui.destroy,bg='red').grid(row=3,column=1,sticky=SE)
    Button2=Button(gui, text="Info",width=6,height=1,font="none 16 bold", command=lambda:abt(),bg='white').grid(row=3,column=1,sticky=SW)
    menu=Menu(gui)
    gui.config(menu=menu)
    def exitt():
        exit()
    def abt():
        tk.messagebox.showinfo("Katherlab.ai application info", "1.Tessellation: This is a process of exctracting user specified smaller tiles from the whole slide images \n2. Normalization: This is a processs of normalizing the tiles before training the model \n3. Training: This process trains the normalized tiles using a deep learning model")
    subm1=Menu(menu)
    menu.add_cascade(label='File', menu=subm1)
    subm1.add_command(label="Exit", command=exitt)

    subm2=Menu(menu)
    menu.add_cascade(label='Option', menu=subm2)
    subm2.add_command(label="About", command=abt)

    gui.mainloop()

def tessellation():
    global gui_1
    ######################################################
    Image.MAX_IMAGE_PIXELS = 100000000000
    args_numberOfThreads = 8
    DEFAULT_JPG_MPP = 0.2494
    JSON_ANNOTATION_SCALE = 10

    ######################################################
    def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    #Initalize the app and provide info about the size of the window and display image
    gui.geometry("5x5")
    gui_1= Toplevel(gui)
    gui_1.title("Tiling process information")
    #pop.geometry("503x400")
    gui_1.geometry("495x600")
    gui_1.resizable(width=0, height=0)
    gui_1.config(bg="black")
    gui_1.title("Kather.ai application for tiling WSI")
    photo_1 = PhotoImage(file=resource_path('images\\display.GIF'))
    Label(gui_1, image=image1, bg="white", fg="white", ).grid(row=0, column=0)
    menu=Menu(gui_1)
    gui_1.config(menu=menu)

    def exitt():
        exit()
    def abt():
        tk.messagebox.showinfo("Katherlab.ai application info", "1.Enter the integer value of thread: Enter the value of how many CPU threads you would like to use for the tiling process\n 2.Enter the patch   value  in  pixel: Enter how many pixels size you want to tile the WSI\n 3.Enter the patch value in micron: Enter the value of how much magnification value in microns")

    subm1=Menu(menu)
    menu.add_cascade(label='File', menu=subm1)
    subm1.add_command(label="Exit", command=exitt)

    subm2=Menu(menu)
    menu.add_cascade(label='Option', menu=subm2)
    subm2.add_command(label="About", command=abt)

    # Class for folder selection and returning the path
    class FolderSelect(Frame):
        def __init__(self,parent=None,folderDescription="",**kw):
            Frame.__init__(self,master=parent,**kw)
            self.folderPath = StringVar()
            self.lblName = Label(self, text=folderDescription, bg="black", fg="white", font="none 10 bold")
            self.lblName.grid(row=0,column=0)
            self.entPath = Entry(self, textvariable=self.folderPath)
            self.entPath.grid(row=0,column=1)
            self.btnFind = ttk.Button(self,text="Browse Folder",command=self.setFolderPath)
            self.btnFind.grid(row=0,column=2)
        def setFolderPath(self):
            folder_selected = filedialog.askdirectory()
            self.folderPath.set(folder_selected)
        @property
        def folder_path(self):
            return self.folderPath.get()

    # Function to display the selected values
    def message_box():
        global pop, lable1
        pop= Toplevel(gui_1)
        gui_1.geometry("5x5")
        pop.title("Tiling process information")
        #pop.geometry("503x400")
        pop.geometry("503x550")
        pop.resizable(width=0, height=0)
        pop.config(bg="black")
        pop_lable= Label(pop,text="Please confirm the entered information!",bg="black", fg="Green",font="none 12 bold")
        pop_lable.grid(row=0, column=0,sticky=W)
        pop_lable1= Label(pop,text="Your entered datails are as follows:",bg="black", fg="white",font="none 12 bold")
        pop_lable1.grid(row=2, column=0,sticky=W)
        Label(pop, text="Entered value of thread:"+ str(data[0][0]), bg="black", fg="white", font="none 12 bold").grid(row=3, column=0,sticky=W)
        Label(pop, text="Entered patch value in pixel:"+ str(data[0][1]), bg="black", fg="white", font="none 12 bold").grid(row=4, column=0,sticky=W)
        Label(pop, text="Entered patch value in micron:"+ str(data[0][2]), bg="black", fg="white", font="none 12 bold").grid(row=5, column=0,sticky=W)
        Label(pop, text="Path for the WSI used to tile:"+ str(data[0][3]), bg="black", fg="white", font="none 12 bold").grid(row=6, column=0,sticky=W)
        Label(pop, text="Path for saving the tiled images:"+ str(data[0][4]), bg="black", fg="white", font="none 12 bold").grid(row=7, column=0,sticky=W)
        Label(pop, text="With annotation then skip WSI:"+ str(data[0][5]), bg="black", fg="white", font="none 12 bold").grid(row=8, column=0,sticky=W)
        Label(pop, text="Augmentation:"+ str(data[0][6]), bg="black", fg="white", font="none 12 bold").grid(row=9, column=0,sticky=W)
        Label(pop, text="Total slides for tiling : "+ str(data[0][7]), bg="black", fg="white", font="none 12 bold").grid(row=10, column=0,sticky=W)
        if overlap == 1:
                #print(data[0][10])
                Label(pop, text="Overlap : "+ str((1-overlap_percent_1)*100)+"%", bg="black", fg="white", font="none 12 bold").grid(row=11, column=0,sticky=W)
        else:
                Label(pop, text="Overlap : "+ str(data[0][10])+"%", bg="black", fg="white", font="none 12 bold").grid(row=11, column=0,sticky=W)
        #Label(pop, text="Slides remailing:", bg="black", fg="white", font="none 12 bold").grid(row=11, column=0,sticky=W)
        Button1=Button(pop, text="Start",width=12,height=2,font="none 16 bold", command=lambda:MainFunction(), justify="left",bg='green').grid(row=15,column=0)
        Button2=Button(pop, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=16,column=0)
        #Button3=Button(pop, text="Tileing Info",width=12,height=1,font="none 16 bold", command=lambda: file_count(), justify="left",bg='green').grid(row=17,column=0)
       

    #B1 = Tkinter.Button(top, text = "Say Hello", command = hello)

    def doStuff():
        global values, data,folder1,folder2, overlap, overlap_percent_1
        data=[]
        folder1 = directory1Select.folder_path
        folder2 = directory2Select.folder_path
        thread= int(textentry.get())
        pixel= int(textentry1.get())
        micron = int(textentry2.get())
        Skip_WSI = var1.get()
        aug = var2.get()
        overlap=var3.get()
        overlap_2= 0
        if Skip_WSI == 1:
            Skip_WSI = True
        else:
            Skip_WSI = False
        if aug == 1:
            augmentation = True
        else:
            augmentation = False
        if overlap == 1:
                overlap = True
                overlap_percent=variable.get()
                overlap_percent_1=float(((100-int(overlap_percent))/100))
                #print(overlap_percent_1)
        else:
                overlap = False
                overlap_2= 0
                overlap_percent_1=float(1.0)
        tot = 0
        #tot2=0
        for root, dirs, files in os.walk(folder1):
         tot += len(files)
        values= thread, pixel, micron, folder1, folder2,Skip_WSI, augmentation, tot,overlap_percent_1, overlap,overlap_2
        data.append(values)
        #gui.exit()
        return(data)



    #Labels and button design for GUI
    directory1Select = FolderSelect(gui_1,"Select folder from where WSI used for tiling:")
    directory1Select.grid(row=11,sticky=W)

    directory2Select = FolderSelect(gui_1,"Select folder to save the images  after tiling:")
    directory2Select.grid(row=12,sticky=W)

    Label(gui_1, text="Enter the integer value of thread:", bg="black", fg="white", font="none 10 bold").grid(row=5, column=0,sticky=W)
    v = StringVar(gui_1, value='8')
    textentry = Entry(gui_1,textvariable=v,width=6, bg="white")
    textentry.place(x=215,y=240)

    Label(gui_1, text="Enter the patch   value  in  pixel:", bg="black", fg="white", font="none 10 bold").grid(row=7, column=0,sticky=W)
    v1 = StringVar(gui_1, value='512')
    textentry1 = Entry(gui_1,textvariable=v1,width=6, bg="white")
    textentry1.place(x=215,y=262)

    Label(gui_1, text="Enter the patch  value in micron:", bg="black", fg="white", font="none 10 bold").grid(row=9, column=0,sticky=W)
    textentry2 = Entry(gui_1,width=6, bg="white")
    v2 = StringVar(gui_1, value='256')
    textentry2 = Entry(gui_1,textvariable=v2,width=6, bg="white")
    textentry2.place(x=215,y=285)

    var1=IntVar()
    Checkbutton(gui_1, text= "With annotation then skip WSI", variable=var1,bg="black",fg="green",font="none 10 bold").grid(row=13,column=0,sticky=W)
    var2=IntVar()
    Checkbutton(gui_1, text= "Augmentation", variable=var2,  bg="black", fg="green",font="none 10 bold").grid(row=14,column=0,sticky=W)
    var3=IntVar()
    Checkbutton(gui_1, text= "Overlap: If overlap is checked select the overlap percentage:", variable=var3,  bg="black", fg="green",font="none 10 bold").grid(row=15,column=0,sticky=W)
    variable = StringVar(gui_1)
    OPTIONS = ['0.25','0.5','0.75','1.0']
    variable.set(OPTIONS[0]) # default value
    w = OptionMenu(gui_1, variable, *OPTIONS).place(x=415,y=402)
    Button1=Button(gui_1, text="Next",width=12,height=2,font="none 16 bold", command=lambda: results(doStuff()), justify="left",bg='green').grid(row=16,column=0)
    #Button(gui, text="Clear",width=6, command= clear_function(textentry,textentry1,textentry2)).grid(row=16,column=0)
    #Button2=Button(gui_1, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=17,column=0)
    Button2=Button(gui_1, text="Quit",width=6,height=1,font="none 16 bold",command=gui.destroy,bg='red').place(x=246,y=500)
    Button3=Button(gui_1, text="Info",width=6,height=1,font="none 16 bold", command=lambda:abt(),bg='white').place(x=163,y=500)
    def results(data):
        message_box()
        #MainFunction()
        

    ####################
    Image.MAX_IMAGE_PIXELS = 100000000000
    NUM_THREADS = 8
    DEFAULT_JPG_MPP = 0.2494
    JSON_ANNOTATION_SCALE = 10

    ####################

    class AnnotationObject:
        def __init__(self, name):
            self.name = name
            self.coordinates = []

        def add_coord(self, coord):
            self.coordinates.append(coord)

        def scaled_area(self, scale):
            return np.multiply(self.coordinates, 1/scale)

        def print_coord(self):
            for c in self.coordinates:
                print(c)

        def add_shape(self, shape):
            for point in shape:
                self.add_coord(point)


    class JPGSlide:
        def __init__(self, path, mpp):
            self.loaded_image = imageio.imread(path)
            self.dimensions = (
                self.loaded_image.shape[1], self.loaded_image.shape[0])
            self.properties = {ops.PROPERTY_NAME_MPP_X: mpp}
            self.level_dimensions = [self.dimensions]
            self.level_count = 1

        def get_thumbnail(self, dimensions):
            return cv2.resize(self.loaded_image, dsize=dimensions, interpolation=cv2.INTER_CUBIC)

        def read_region(self, topleft, level, window):
            return self.loaded_image[topleft[1]:topleft[1] + window[1],
                                     topleft[0]:topleft[0] + window[0], ]


    class SlideReader:
        def __init__(self, path, filetype, export_folder = None, pb = None):
            
            self.coord = []
            self.annotations = []
            self.export_folder = export_folder
            self.pb = pb  # Progress bar
            self.p_id = None
            self.extract_px = None
            self.shape = None
            self.basename = path.replace('.'+ path.split('.')[-1],'')
            self.name = self.basename.split('\\')[-1]
            self.has_anno = True
            self.annPolys = []
            self.ignoredFiles = []
            self.noMPPFlag = 0
            self.IsLoadedCorrectly = False
            if filetype in ["svs", "mrxs", 'ndpi', 'scn', 'tif']:
                try:
                    self.slide = ops.OpenSlide(path)
                except:
                    outputFile.write('Unable to read ' + filetype + ',' + path + '\n')
                    self.IsLoadedCorrectly = True
                    return None
            elif filetype == "jpg":            
                self.slide = JPGSlide(path, mpp=DEFAULT_JPG_MPP)
            else:
                outputFile.write('Unsupported file type ' + filetype + ',' + path + '\n')
                return None

            thumbs_path = join(export_folder, "thumbs")
            if not os.path.exists(thumbs_path):
                os.makedirs(thumbs_path)
                
            # Load ROIs if available
            roi_path_csv = self.basename + ".csv"
            roi_path_json = self.basename + ".json"

            if exists(roi_path_csv) and not os.path.getsize(roi_path_csv) == 0:
                self.load_csv_roi(roi_path_csv)
            elif exists(roi_path_json) and not os.path.getsize(roi_path_json) == 0:
                self.load_json_roi(roi_path_json)
            else:
                self.has_anno = False
                
            self.shape = self.slide.dimensions
            self.filter_dimensions = self.slide.level_dimensions[-1]
            self.filter_magnification = self.filter_dimensions[0] / self.shape[0]
            goal_thumb_area = 4096 * 4096
            y_x_ratio = self.shape[1] / self.shape[0]
            thumb_x = sqrt(goal_thumb_area / y_x_ratio)
            thumb_y = thumb_x * y_x_ratio
            self.thumb = self.slide.get_thumbnail((int(thumb_x), int(thumb_y)))
            self.thumb_file = thumbs_path + '/' + self.name + '_thumb.jpg'       
            imageio.imwrite(self.thumb_file, self.thumb)
            try:
                self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
            except:
                self.noMPPFlag = 1
                outputFile.write('No PROPERTY_NAME_MPP_X' + ',' + path + '\n')
                return None
                
        def loaded_correctly(self):
            return bool(self.shape)

        def build_generator(self, size_px, size_um, stride_div, case_name, tiles_path, category, fileSize, export=False, augment=False, normalization = False, normalization_sample = ''):
                        
            self.extract_px = int(size_um / self.MPP)
            stride = int(self.extract_px * stride_div)
            
            slide_x_size = self.shape[0] - self.extract_px
            slide_y_size = self.shape[1] - self.extract_px
            
            for y in range(0, (self.shape[1]+1) - self.extract_px, stride):
                for x in range(0, (self.shape[0]+1) - self.extract_px, stride):
                    is_unique = ((y % self.extract_px == 0) and (x % self.extract_px == 0))
                    self.coord.append([x, y, is_unique])

            self.annPolys = [sg.Polygon(annotation.coordinates) for annotation in self.annotations]
            
            tile_mask = np.asarray([0 for i in range(len(self.coord))])
            self.tile_mask = None
            
            def generator():
                for ci in range(len(self.coord)):
                    c = self.coord[ci]
                    filter_px = int(self.extract_px * self.filter_magnification)
                    if filter_px == 0:
                        filter_px = 1
                    # Check if the center of the current window lies within any annotation; if not, skip
                    if bool(self.annPolys) and not any([annPoly.contains(sg.Point(int(c[0]+self.extract_px/2), int(c[1]+self.extract_px/2))) for annPoly in self.annPolys]):
                        continue
                    
                    # Read the low-mag level for filter checking
                    filter_region = np.asarray(self.slide.read_region(c, self.slide.level_count-1, [filter_px, filter_px]))[:, :, :-1]
                    median_brightness = int(sum(np.median(filter_region, axis=(0, 1))))
                    if median_brightness > 660:
                        continue
                    elif median_brightness == 0:
                        continue

                    # Read the region and discard the alpha pixels
                    try:
                        region = np.asarray(self.slide.read_region(c, 0, [self.extract_px, self.extract_px]))[:, :, 0:3]
                        region = cv2.resize(region, dsize=(size_px, size_px), interpolation=cv2.INTER_CUBIC)
                    except:
                        continue
                                    
                    b, g, r    = cv2.split(region) # For BGR image                
                    standardDeviation = np.std([np.mean(b), np.mean(g), np.mean(r)])
                    edge  = cv2.Canny(region, 40, 40) 
                    edge = edge / np.max(edge)
                    edge = (np.sum(np.sum(edge)) / (size_px *size_px)) * 100
                    if (standardDeviation < 1) or (standardDeviation < 2 and edge < 5) or (standardDeviation > 20 and edge < 5) or (edge < 4) or np.isnan(edge):                                                               
                        continue 

                    if normalization:
                        region = normalization_sample.transform(region)
                        
                    tile_mask[ci] = 1
                    coord_label = ci
                    unique_tile = c[2]
                    
                    if export and unique_tile:                     
                        imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+').jpg'), region)
                        if augment:
                            imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+')._aug1.jpg'), np.rot90(region))
                            imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+')._aug2.jpg'), np.flipud(region))
                            imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+')._aug3.jpg'), np.flipud(np.rot90(region)))
                            imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+')._aug4.jpg'), np.fliplr(region))
                            imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+')._aug5.jpg'), np.fliplr(np.rot90(region)))
                            imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+')._aug6.jpg'), np.flipud(np.fliplr(region)))
                            imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+')._aug7.jpg'), np.flipud(np.fliplr(np.rot90(region))))
                    yield region, coord_label, unique_tile

                if self.pb:
                    if sum(tile_mask) <4:
                        outputFile.write('Number of Extracted Tiles < 4 ' + ',' + join(tiles_path, case_name)+ '\n')
                    
                    print('Remained Slides: ' + str(fileSize))
                    print('***************************************************************************')
                        
                self.tile_mask = tile_mask

            return generator, slide_x_size, slide_y_size, stride

        def load_csv_roi(self, path):
            reader = csv.DictReader(path)
            headers = []
            for col in reader.columns: 
                headers.append(col.strip()) 
            if 'X_base' in headers and 'Y_base' in headers:
                index_x = headers.index('X_base')
                index_y = headers.index('Y_base')
            else:
                raise IndexError('Unable to find "X_base" and "Y_base" columns in CSV file.')            
            self.annotations.append(AnnotationObject("Object" + str(len(self.annotations))))
            for index, row in reader.iterrows():
                if(str(row[index_x]).strip() == 'X_base' or str(row[index_y]).strip() == 'Y_base'):
                    self.annotations.append(AnnotationObject(f"Object{len(self.annotations)}"))
                    continue
                
                x_coord = int(float(row[index_x]))
                y_coord = int(float(row[index_y]))           
                self.annotations[-1].add_coord((x_coord, y_coord))
                
        def load_json_roi(self, path):
            with open(path, "r") as json_file:
                json_data = json.load(json_file)['shapes']
            for shape in json_data:
                area_reduced = np.multiply(shape['points'], JSON_ANNOTATION_SCALE)
                self.annotations.append(AnnotationObject("Object" + len(self.annotations)))
                self.annotations[-1].add_shape(area_reduced)
            
    class Convoluter:
        def __init__(self, size_px, size_um, stride_div,  num_classes, batch_size, use_fp16, save_folder='', normalization_sample = '', skipws=False, augment=False, normalization = False):
            
            self.SLIDES = {}
            self.MODEL_DIR = None
            self.PKL_DICT = {}
            self.SIZE_PX = size_px
            self.SIZE_UM = size_um
            self.NUM_CLASSES = num_classes        
            self.BATCH_SIZE = batch_size
            self.SAVE_FOLDER = save_folder
            self.STRIDE_DIV = stride_div
            self.MODEL_DIR = None
            self.AUGMENT = augment
            self.skipws = skipws
            self.normalization = normalization
            self.normalization_sample = normalization_sample
            
        def load_slides(self, slides_array, directory = "None", category = "None"):
            self.fileSize = len(slides_array)
            self.iterator = 0
            print('TOTAL NUMBER OF SLIDES IN THIS FOLDER : ' + str(self.fileSize))
            
            for slide in slides_array:
                name = slide.split('.')[:-1]
                name ='.'.join(name)
                name = name.split('\\')[-1]
                filetype = slide.split('.')[-1]
                path = slide

                self.SLIDES.update({name: {"name": name,
                                           "path": path,
                                           "type": filetype,
                                           "category": category}})
        
            return self.SLIDES

        def convolute_slides(self):
            '''Parent function to guide convolution across a whole-slide image and execute desired functions.
            '''
            ignoredFile_list = []
            if not os.path.exists(join(self.SAVE_FOLDER, "BLOCKS")):
                os.makedirs(join(self.SAVE_FOLDER, "BLOCKS"))
            # self.STRIDE_DIV=1
            pb = progressbar.ProgressBar()
            pool = ThreadPool(NUM_THREADS)
            pool.map(lambda slide: self.export_tiles(self.SLIDES[slide], pb, ignoredFile_list), self.SLIDES)
            return pb, ignoredFile_list

        def export_tiles(self, slide, pb, ignoredFile_list):
            case_name = slide['name']
            category = slide['category']
            path = slide['path']
            filetype = slide['type']
            self.iterator = self.iterator + 1
            whole_slide = SlideReader(path, filetype, self.SAVE_FOLDER, pb=pb)

            if not whole_slide.has_anno and self.skipws:
                #print("Did not find .csv file, skipping " + filetype)
                return
            
            if whole_slide.IsLoadedCorrectly:
                return
            
            if whole_slide.noMPPFlag:
                return
            
            tiles_path = whole_slide.export_folder + '/' + "BLOCKS"
            if not os.path.exists(tiles_path):
                os.makedirs(tiles_path)
                
            tiles_path = tiles_path + '/' + case_name
                
               
            if not os.path.exists(tiles_path):
                os.makedirs(tiles_path)
                
            counter = len(os.listdir(tiles_path))
            if counter > 6:
               #print("Folder already filled")
               #print('***************************************************************************')
               return  
            
            gen_slice, _, _, _ = whole_slide.build_generator(self.SIZE_PX, self.SIZE_UM, self.STRIDE_DIV, case_name, tiles_path, category, 
                                                             fileSize = self.fileSize - self.iterator, export=True,
                                                             augment=self.AUGMENT, normalization = self.normalization, normalization_sample = self.normalization_sample)
            for tile, coord, unique in gen_slice():
                pass

    def MainFunction():
        global pop2
        display_count = data[0][7]
        args_out = str(data[0][4])
        args_numberOfThreads = data[0][0]
        args_px = data[0][1]
        args_um = data[0][2]
        args_classes = 1
        args_batch = 512
        args_fp16 = True
        args_augment = data[0][6]
        args_skipws = data[0][5]
        args_slide = str(data[0][3])
        ags_ov = data[0][8]

        c = Convoluter(args_px, args_um,ags_ov, args_classes, args_batch,args_fp16, args_out, augment = args_augment, skipws = args_skipws)
        global outputFile
        outputFile  = open(os.path.join(args_out,'report.txt'), 'a',encoding="utf-8")
        
        if isfile(args_slide):
            path_sep = os.path.sep
            slide_list = [args_slide.split('/')[-1]]
            slide_dir = '/'.join(args_slide.split('/')[:-1])
            c.load_slides(slide_list, slide_dir)
        else:
            
            slide_list = []        
            for root, dirs, files in os.walk(args_slide):
                for file in files:
                    if ('.ndpi' in file or '.scn' in file or 'svs' in file or 'tif' in file) and not 'csv' in file:
                        slide_list.append(os.path.join(root, file))
       
            if os.path.exists(join(args_out, "BLOCKS")):
                temp = os.listdir(os.path.join(args_out, 'BLOCKS'))                         
                for item in temp:
                    for s in slide_list:
                        if item in s:
                            slide_list.remove(s)
            c.load_slides(slide_list)        
        pb = c.convolute_slides()
        print('this is the end')
        process = 10
        if process==10:
            pop2 = Toplevel(pop)
            pop.geometry("5x5")
            pop2.title("Tiling process information")
            pop2.geometry("325x250")
            pop2.resizable(width=0, height=0)
            pop2.config(bg="black")    
            pop2_lable= Label(pop2,text="Tiling process completed!!!",bg="black", fg="Green",font="none 18 bold")
            pop2_lable.grid(row=2, column=0,sticky=W)
            Label(pop2, text="Total slides tiled : "+ str(data[0][7]), bg="black", fg="white", font="none 12 bold").grid(row=3, column=0,sticky=W)
            #Button1=Button(pop2, text="Main menu",width=12,height=2,font="none 16 bold", command=lambda:opening_page(), justify="left",bg='green').grid(row=15,column=0)
            Button2=Button(pop2, text="Normalization",width=12,height=2,font="none 16 bold", command=lambda:normalization(), justify="left",bg='green').grid(row=16,column=0)
            Button3=Button(pop2, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=17,column=0)
        else:
            pop2 = Toplevel(pop)
            pop.geometry("5x5")
            pop2.title("Tiling process information")
            pop2.geometry("250x250")
            pop2.resizable(width=0, height=0)
            pop2.config(bg="black")    
            pop2_lable= Label(pop2,text="Tiling process not completed!!!",bg="black", fg="Red",font="none 18 bold")
            pop2_lable.grid(row=2, column=0,sticky=W)
            #Button1=Button(pop, text="Main menu",width=12,height=2,font="none 16 bold", command=lambda:opening_page(), justify="left",bg='green').grid(row=15,column=0)
            #Button2=Button(pop, text="Normalization",width=12,height=2,font="none 16 bold", command=lambda:normalization(), justify="left",bg='green').grid(row=16,column=0)
            Button3=Button(pop, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=17,column=0)
        outputFile.close()
##############################################################################################################################################################################################################################################################

def normalization():
    
    gui.geometry("5x5")
    gui_2= Toplevel(gui)
    gui_2.geometry("670x500")
    gui_2.resizable(width=0, height=0)
    gui_2.config(bg="black")
    gui_2.title("Kather.ai application for normalization")
    Label(gui_2, image=image1, bg="white", fg="white",).grid(row=0, column=0)
    menu=Menu(gui_2)
    gui_2.config(menu=menu)
    
    def exitt():
        exit()
    def abt():
        tk.messagebox.showinfo("Katherlab.ai application info", "1.Enter the integer value of thread: Enter the value of how many CPU threads you would like to use for the normalization process\n 2. Enter method: Select the method you want to chose for normalization")

    subm1=Menu(menu)
    menu.add_cascade(label='File', menu=subm1)
    subm1.add_command(label="Exit", command=exitt)

    subm2=Menu(menu)
    menu.add_cascade(label='Option', menu=subm2)
    subm2.add_command(label="About", command=abt)
    
    class FolderSelect(Frame):
        def __init__(self,parent=None,folderDescription="",**kw):
            Frame.__init__(self,master=parent,**kw)
            self.folderPath = StringVar()
            self.lblName = Label(self, text=folderDescription, bg="black", fg="white", font="none 12 bold")
            self.lblName.grid(row=0,column=0)
            self.entPath = Entry(self, textvariable=self.folderPath)
            self.entPath.grid(row=0,column=1)
            self.btnFind = ttk.Button(self,text="Browse Folder",command=self.setFolderPath)
            self.btnFind.grid(row=0,column=2)
        def setFolderPath(self):
            folder_selected = filedialog.askdirectory()
            self.folderPath.set(folder_selected)
        @property
        def folder_path(self):
            return self.folderPath.get()

    def message_box():
        global pop, lable1
        gui_2.geometry("5x5")
        pop= Toplevel(gui_2)
        pop.title("Normalization process information")
        #pop.geometry("503x400")
        pop.geometry("503x500")
        pop.resizable(width=0, height=0)
        pop.config(bg="black")
        pop_lable= Label(pop,text="Please confirm the entered information!",bg="black", fg="Green",font="none 12 bold")
        pop_lable.grid(row=0, column=0,sticky=W)
        pop_lable1= Label(pop,text="Your entered datails are as follows:",bg="black", fg="white",font="none 12 bold")
        pop_lable1.grid(row=2, column=0,sticky=W)
        Label(pop, text="Entered value of thread:"+ str(data[0][0]), bg="black", fg="white", font="none 12 bold").grid(row=3, column=0,sticky=W)
        Label(pop, text="Entered path for tiles to normalize :"+ str(data[0][1]), bg="black", fg="white", font="none 12 bold").grid(row=4, column=0,sticky=W)
        Label(pop, text="Entered path to save the normalized tiles :"+ str(data[0][2]), bg="black", fg="white", font="none 12 bold").grid(row=5, column=0,sticky=W)
        Label(pop, text="Entered path to save the faile to normalized tiles:"+ str(data[0][3]), bg="black", fg="white", font="none 12 bold").grid(row=6, column=0,sticky=W)
        Label(pop, text="Method selected: "+ str(data[0][4]), bg="black", fg="white", font="none 12 bold").grid(row=7, column=0,sticky=W)
        #Label(pop, text="Skip WSI:"+ str(data[0][5]), bg="black", fg="white", font="none 12 bold").grid(row=8, column=0,sticky=W)
        #Label(pop, text="Augmentation:"+ str(data[0][6]), bg="black", fg="white", font="none 12 bold").grid(row=9, column=0,sticky=W)
        Label(pop, text="Total slides for tileing : "+ str(data[0][5]), bg="black", fg="white", font="none 12 bold").grid(row=10, column=0,sticky=W)
        #Label(pop, text="Slides remailing:", bg="black", fg="white", font="none 12 bold").grid(row=11, column=0,sticky=W)
        Button1=Button(pop, text="Start",width=12,height=2,font="none 16 bold", command=lambda:MainFunction(), justify="left",bg='green').grid(row=15,column=0)
        Button2=Button(pop, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=16,column=0)

    def doStuff():
        global data,values,thread, data,folder1,folder2, folder3, method
        data=[]
        folder1 = directory1Select.folder_path
        folder2 = directory2Select.folder_path
        folder3 = directory3Select.folder_path
        thread= int(textentry.get())
        method=variable.get() 
        tot = 0
        #tot2=0
        for root, dirs, files in os.walk(folder1):
         tot += len(files)
        values= thread, folder1, folder2,folder3,method, tot
        data.append(values)
        #gui.exit()
        return(data)

    #Labels and button design for GUI
    directory1Select = FolderSelect(gui_2,"Select folder from where tiles are selected for normalization:")
    directory1Select.grid(row=11,sticky=W)

    directory2Select = FolderSelect(gui_2,"Select folder  to save the  normalized  tiles after the process:")
    directory2Select.grid(row=12,sticky=W)

    directory3Select = FolderSelect(gui_2,"Select  folder  to  save the  failed  tiles  during  normalization:")
    directory3Select.grid(row=13,sticky=W)

    Label(gui_2, text="Enter the integer value of thread:", bg="black", fg="white", font="none 12 bold").grid(row=5, column=0,sticky=W)
    v = StringVar(gui, value='8')
    textentry = Entry(gui_2,textvariable=v,width=6, bg="white")
    textentry.place(x=252,y=240)

    Label(gui_2, text="Select the normalization method:", bg="black", fg="white", font="none 12 bold").grid(row=6, column=0,sticky=W)
    OPTIONS = ['stainNorm_Reinhard','stainNorm_Macenko','stainNorm_Vahadane']
    variable = StringVar(gui_2)
    variable.set(OPTIONS[0]) # default value
    w = OptionMenu(gui_2, variable, *OPTIONS).place(x=252,y=260)                       
    Button1=Button(gui_2, text="Next",width=12,height=2,font="none 16 bold", command=lambda: results(doStuff()), justify="left",bg='green').place(x=253,y=375)
    #Button(gui, text="Clear",width=6, command= clear_function(textentry,textentry1,textentry2)).grid(row=16,column=0)
    #Button2=Button(gui_2, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=17,column=0)
    Button2=Button(gui_2, text="Quit",width=6,height=1,font="none 16 bold",command=gui.destroy,bg='red').place(x=331,y=440)
    Button3=Button(gui_2, text="Info",width=6,height=1,font="none 16 bold", command=lambda:abt(),bg='white').place(x=253,y=440)
    def results(data):
        message_box()
        
    def Normalize(item):
        outputPathRoot = os.path.join(outputPath, item)
        inputPathRoot = os.path.join(inputPath, item)
        inputPathRootContent = os.listdir(inputPathRoot)
        if os.path.exists(outputPathRoot):
            outputPathRootContent =  os.listdir(outputPathRoot) 
        else:
            outputPathRootContent = []
        if not os.path.exists(outputPathRoot):
            os.mkdir(outputPathRoot)
        if os.path.exists(outputPathRoot) and not len(outputPathRootContent) == len(inputPathRootContent):
            temp = os.path.join(inputPath, item)
            tempContent = os.listdir(temp)
            tempContent = [i for i in tempContent if i.endswith('.jpg')]
            print(temp)
            time.sleep(5)
            for tempItem in tempContent:
                img = cv2.imread(os.path.join(inputPathRoot, tempItem))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                edge  = cv2.Canny(img, 40, 40) 
                edge = edge / np.max(edge)           
                edge = (np.sum(np.sum(edge)) / (imageSize *imageSize)) * 100
                if edge>2:
                    try:
                        nor_img = n.transform(img)
                        cv2.imwrite(os.path.join(outputPathRoot, tempItem), cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))
                    except:
                        cv2.imwrite(os.path.join(removePath, tempItem), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(os.path.join(removePath, tempItem), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def MainFunction():
        global imageSize,pop2, inputPath, outputPath, removePath, n
        inputPath = str(folder1)
        outputPath = str(folder2)
        removePath = str(folder3)
        print('input path is '+ inputPath)
        print('output path is '+ outputPath)
        print('failed path is '+ removePath)
        #method=data[0][4]
        inputPathContent = os.listdir(inputPath)
        inputPathContent = [i for i in inputPathContent if not i.endswith('.bat')]
        inputPathContent = [i for i in inputPathContent if not i.endswith('.txt')]
        path=resource_path('images\\source.jpg')
        target = cv2.imread(path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        imageSize = target.shape[0]
        if method == 'stainNorm_Reinhard':
            n = stainNorm_Reinhard.Normalizer()
            n.fit(target)
            threads = thread
            pool = ThreadPool(threads)
            pool.map(Normalize, inputPathContent) 
            pool.close()
            pool.join()
            process= 10
        elif method == 'stainNorm_Macenko':
            n = stainNorm_Macenko.Normalizer()
            n.fit(target)
            threads = thread
            pool = ThreadPool(threads)
            pool.map(Normalize, inputPathContent)
            pool.close()
            pool.join()
            process= 10
        elif method =='stainNorm_Vahadane':
            n = stainNorm_Vahadane.Normalizer()
            n.fit(target)
            threads = thread
            pool = ThreadPool(threads)
            pool.map(Normalize, inputPathContent) 
            pool.close()
            pool.join()
            process= 10
        else:
            print('Error in method selected')
            process=20
        if process == 10:
            pop2 = Toplevel(pop)
            pop.geometry("5x5")
            pop2.title("Normalization process information")
            pop2.geometry("250x250")
            pop2.resizable(width=0, height=0)
            pop2.config(bg="black")    
            Label(pop2,text="Process completed!!!",bg="black", fg="Green",font="none 18 bold").grid(row=3, column=0,sticky=W)
            #Button1=Button(pop2, text="Main menu",width=12,height=2,font="none 16 bold", command=lambda:opening_page(), justify="left",bg='green').grid(row=15,column=0)
            Button2=Button(pop2, text="Training",width=12,height=2,font="none 16 bold", command=lambda:training(), justify="left",bg='green').grid(row=16,column=0)
            Button3=Button(pop2, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=17,column=0)
        else:
            pop2 = Toplevel(pop)
            pop.geometry("5x5")
            pop2.title("Normalization process information")
            pop2.geometry("250x250")
            pop2.resizable(width=0, height=0)
            pop2.config(bg="black")    
            Label(pop2,text="Process not completed!!!",bg="black", fg="Red",font="none 18 bold").grid(row=3, column=0,sticky=W)
            #Button1=Button(pop2, text="Main menu",width=12,height=2,font="none 16 bold", command=lambda:opening_page(), justify="left",bg='green').grid(row=15,column=0)
            #Button2=Button(pop2, text="Training",width=12,height=2,font="none 16 bold", command=lambda:training(), justify="left",bg='green').grid(row=16,column=0)
            Button3=Button(pop2, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=17,column=0)

##################################################
def training():
    class FolderSelect(Frame):
        def __init__(self,parent=None,folderDescription="",**kw):
            Frame.__init__(self,master=parent,**kw)
            self.folderPath = StringVar()
            self.lblName = Label(self, text=folderDescription, bg="black", fg="white", font="none 10 bold")
            self.lblName.grid(row=0,column=0)
            self.entPath = Entry(self, textvariable=self.folderPath)
            self.entPath.grid(row=0,column=1)
            self.btnFind = ttk.Button(self,text="Browse Folder",command=self.setFolderPath)
            self.btnFind.grid(row=0,column=2)
        def setFolderPath(self):
            folder_selected = filedialog.askdirectory()
            self.folderPath.set(folder_selected)
        @property
        def folder_path(self):
            return self.folderPath.get()

        
    ###############################################################################

    def message_box():
        global pop, lable1
        gui_2.geometry("5x5")
        pop= Toplevel(gui_2)
        pop.title("Thumb extraction process information")
        #pop.geometry("503x400")
        pop.geometry("900x300")
        pop.resizable(width=0, height=0)
        pop.config(bg="black")
        pop_lable= Label(pop,text="Please confirm the entered information!",bg="black", fg="Green",font="none 12 bold")
        pop_lable.grid(row=0, column=0,sticky=W)
        pop_lable1= Label(pop,text="Your entered datails are as follows:",bg="black", fg="white",font="none 12 bold")
        pop_lable1.grid(row=2, column=0,sticky=W)
        #Label(pop, text="Entered value of thread:"+ str(data[0][0]), bg="black", fg="white", font="none 12 bold").grid(row=3, column=0,sticky=W)
        Label(pop, text="Entered path for tiles to extract thumb files :"+ str(data[0][0]), bg="black", fg="white", font="none 12 bold").grid(row=4, column=0,sticky=W)
        Label(pop, text="Entered path to save the thumb tiles :"+ str(data[0][1]), bg="black", fg="white", font="none 12 bold").grid(row=5, column=0,sticky=W)
        Button1=Button(pop, text="Start",width=12,height=2,font="none 16 bold", command=lambda:MainFunction(), justify="left",bg='green').grid(row=15,column=0)
        Button2=Button(pop, text="Quit",width=12,height=2,font="none 16 bold",command=gui.destroy,bg='red').grid(row=16,column=0)
    def results(data):
        message_box()

    def doStuff():
        global data,values,thread, data,folder1,folder2, method
        data=[]
        folder1 = directory1Select.folder_path
        folder2 = directory2Select.folder_path
        
        values= folder1, folder2
        data.append(values)
        #gui.exit()
        return(data)
    gui.geometry("5x5")
    gui_2= Toplevel(gui)
    gui_2.geometry("630x400")
    gui_2.resizable(width=0, height=0)
    gui_2.config(bg="black")
    gui_2.title("Kather.ai application for thumb extraction")
    Label(gui_2, image=image1, bg="white", fg="white",).grid(row=0, column=0)
    #Label(gui_2,text="GUI in progress, training module will be added soon.....",bg="black", fg="Red",font="none 18 bold").grid(row=3, column=0,sticky=W)
    gui.geometry("5x5")
    
    #pop.geometry("503x400")
    directory1Select = FolderSelect(gui_2,"Select folder from where WSI are selected for thumb extraction:")
    directory1Select.grid(row=11,sticky=W)

    directory2Select = FolderSelect(gui_2," Select  folder  to  save  the  thumb  files   after  the  extraction :")
    directory2Select.grid(row=12,sticky=W)
    Button1=Button(gui_2, text="Next",width=12,height=2,font="none 16 bold", command=lambda: results(doStuff()), justify="left",bg='green').place(x=253,y=300)
    Button2=Button(gui_2, text="Quit",width=6,height=1,font="none 16 bold",command=gui.destroy,bg='red').place(x=331,y=365)
    Button3=Button(gui_2, text="Info",width=6,height=1,font="none 16 bold", command=lambda:abt(),bg='white').place(x=253,y=365)


    def MainFunction():
        outputPath = str(folder2)
        inputPath = str(folder1)
        print(folder1,folder2)
        thumb_path = join(outputPath, "thumbs")
        
        global outputFile
        outputFile  = open(os.path.join(outputPath,'report.txt'), 'a', encoding="utf-8")
         
        slide_list = []        
        for root, dirs, files in os.walk(inputPath):
            for file in files:
                if ('.ndpi' in file or '.scn' in file or 'svs' in file or 'tif' in file) and not 'csv' in file:
                    fileType = file.split('.')[-1]
                    slide_list.append(os.path.join(root, file))
       
            if not os.path.exists(join(outputPath, "thumbs")):
                os.makedirs(thumb_path)
                
        for item in slide_list:
            NotAbleToLoad = False
            try:
                slide = ops.OpenSlide(item)
            except:
                outputFile.write('Unable to read ' + item + '\n')
                NotAbleToLoad = True
            if not NotAbleToLoad:
                try:
                    name = Path(item.split('\\')[-1]).stem
                    shape = slide.dimensions
                    goal_thumb_area = 4096 * 4096
                    y_x_ratio = shape[1] / shape[0]
                    thumb_x = sqrt(goal_thumb_area / y_x_ratio)
                    thumb_y = thumb_x * y_x_ratio
                    thumb = slide.get_thumbnail((int(thumb_x), int(thumb_y)))
                    thumb_file = thumb_path + '/' + name + '_thumb.jpg'       
                    imageio.imwrite(thumb_file, thumb)
                except:
                    outputFile.write('Cant Load thumb File' + ',' + item + '\n')
        
opening_page()

