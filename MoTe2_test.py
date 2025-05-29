import argparse
import json
import os

import cv2
import numpy as np

import matplotlib.pyplot as plt

from demo.demo_functions import visualise_flakes
from demo.demo_functions import remove_vignette

from GMMDetector import MaterialDetector

with open("C:/Users/bussi/INRS/2DMatGMM/GMMDetector/contrast_data_2H_oversens.json") as f:

    contrast_dict = json.load(f)


model = MaterialDetector(
    contrast_dict=contrast_dict,
    size_threshold=150,
    standard_deviation_threshold=5,
    used_channels="BGR"
)

flatfield = cv2.imread("C:/Users/bussi/INRS/flatfield.png")

image_names = os.listdir("C:/Users/bussi/INRS/2DMatGMM/Datasets/GMMDetectorDatasets/MoTe2 2H/train_images")

for name in image_names:

    image_path = os.path.join("C:/Users/bussi/INRS/2DMatGMM/Datasets/GMMDetectorDatasets/MoTe2 2H/train_images", name)
    image = cv2.imread(image_path)

    image = remove_vignette(image, flatfield)

    flake = model.detect_flakes(image)

    overlay = visualise_flakes(flake, image, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay[:, :, ::-1])
    plt.axis("off")
    plt.show()