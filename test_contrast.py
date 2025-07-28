from GMMDetector import MaterialDetector
from Image_Analysis import Image_Analysis

#Dont really need it but oh well
path_model_contrast = ""

path_dataset_image = ""
flatfield_path = ""
path_microscope_image = ""

Test = Image_Analysis(path_dataset_image, flatfield_path)

Test.Pre_Process()
Test.get_Mean_Background_Value()


