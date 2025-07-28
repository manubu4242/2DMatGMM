from Image_Analysis import *

model_path = r"C:\Users\bussi\INRS\2DMatGMM\GMMDetector\trained_parameters\Graphene_GMM.json"

flatfield_path = r"C:\Users\bussi\INRS\2DMatGMM\Datasets\GMMDetectorDatasets\WB_calibration_graphene\purple\flatfield.png"

images_path = r"C:\Users\bussi\INRS\2DMatGMM\Datasets\GMMDetectorDatasets\WB_calibration_graphene\purple"


Test = Image_Analysis(images_path, flatfield_path)
Test.Init_GMM_Detector(model_path)
Test.Pre_Process()
Test.get_Mean_Background_Value()

print()
Dataset_path = r"C:\Users\bussi\INRS\2DMatGMM\Datasets\GMMDetectorDatasets\Graphene\train_images\3b2fa580-860c-4eb7-a932-cc0a599dc4ad.jpg"

Test2 = Image_Analysis(Dataset_path, flatfield_path, image_dir=False)
Test2.Init_GMM_Detector(model_path)
Test2.get_Mean_Background_Value()

