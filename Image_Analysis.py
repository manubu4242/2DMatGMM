import cv2
from demo.demo_functions import remove_vignette
from demo.demo_functions import visualise_flakes
import os, glob
import numpy as np
import json
import matplotlib.pyplot as plt


from GMMDetector import MaterialDetector


class Image_Analysis():


    def __init__(self, image_path:str, flatfield_path:str, image_dir = True):

        self.images = []
        self.Processed_images = []

        if image_dir:

            os.chdir(image_path)

            for image in glob.glob("*.png"):

                self.images.append(cv2.imread(image))

        else: 

            self.images.append(cv2.imread(image_path))


        self.flatfield = cv2.imread(flatfield_path)
        

    def Pre_Process(self, max_background_value=241):

        for image in self.images:

            new_img = remove_vignette(image, self.flatfield, max_background_value=max_background_value)
            new_img = cv2.medianBlur(new_img, 5)
            self.Processed_images.append(new_img)
    

    def Init_GMM_Detector(self, contrast_path, size=200, std=5, FP_path=None):

        with open(contrast_path) as f:

            contrast = json.load(f)

        self.model = MaterialDetector(
            contrast_dict=contrast,
            size_threshold=size,
            standard_deviation_threshold=std,
            false_positive_detector_path=FP_path
        )
    

    def get_Mean_Background_Value(self, print_BGR=True):

        self.BGR_Values = []

        if len(self.Processed_images) != 0:
            for image in self.Processed_images:
            
                self.BGR_Values.append(self.model.get_mean_background_values(image))
        
        else:
            for image in self.images:
            
                self.BGR_Values.append(self.model.get_mean_background_values(image))

        
        if print_BGR:

            print(self.BGR_Values)

    
    def Find_and_Print_Flakes(self):

        for image in self.Processed_images:

            flake = self.model.detect_flakes(image)

            overlay = visualise_flakes(flake, image, 0)

            plt.figure(figsize=(10, 10))
            plt.imshow(overlay[:, :, ::-1])
            plt.axis("off")
            plt.show()  