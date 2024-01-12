import numpy as np
from numpy import asarray

from PIL import Image

import os

import glob

from keras.layers import Dense, Softmax, Conv2D, Input, MaxPooling2D, Flatten
from keras.models import Sequential

from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from joblib import dump

def main():
    directory = "./data/Testing"

def convertImages(directory):
    pat = f"{directory}/*.jpg"
    
    fileLs = glob.glob(pat)
    for file in fileLs:
        print(f"Creating {file}")
        image = Image.open(file)
        numpyData = asarray(image)


        fileName = os.path.basename(file)

        saveFile = os.path.join("./data/MixedTrainingNumpy", os.path.splitext(fileName)[0] + ".npy")
        print(saveFile)
        np.save(saveFile, numpyData)