import numpy as np
from numpy import asarray
from PIL import Image
import os
import glob
from keras.layers import Dense, Softmax, Conv2D, Input, MaxPooling2D, Flatten
from keras.models import Sequential, load_model

from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from joblib import dump

class dataSet(Dataset):
    def __init__(self, directory):
        self.filePath = glob.glob(f"{directory}/*.npy")
    
    def __len__(self):
        return len(self.filePath)
    
    def __getitem__(self, index):
        filePath = self.filePath[index]
        image = np.load(filePath)
        image = resize(image, (512, 512), anti_aliasing=True)
        image = image / 255
        label = os.path.basename(filePath)[0]
        imageTensor = torch.from_numpy(image).type(torch.FloatTensor)
        return imageTensor, label

def main():
    directory = "./data/MixedTrainingNumpy"

    data = dataSet(directory)
    loadData = DataLoader(data, batch_size=64, shuffle=True)

    #images, labels = loadImages(directory)

    model = Sequential()

    featureExtractor = dataExtraction(model)
    allFeatures = []
    allLabels = []
    for batchImages, batchLabels in loadData:
        batchImagesNP = batchImages.numpy().reshape(-1, 512, 512, 3)
        features = featureExtractor.predict(batchImagesNP)
        allFeatures.extend(features)
        allLabels.extend(batchLabels)
    
    allFeatureNP = np.array(allFeatures)
    allLabelsNP = np.array(allLabels)
    xTrain, xTest, yTrain, yTest = train_test_split(allFeatureNP, allLabelsNP, test_size=0.2, random_state=42)
    print("worked")

    hyperparamGrid = {
        'n_estimators' : [100, 500, 1000],
        'max_depth' : [10, 20, 30, None],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 3],
        'max_features' : ['auto', 'sqrt'],
        'bootstrap' : [True, False]
    }
    randfModel = RandomForestClassifier() #TODO - actually train random forest, figure out a way to save the model
    hyperparamSearch = GridSearchCV(estimator=randfModel, param_grid=hyperparamGrid, cv=3, n_jobs=-1, verbose=2)
    hyperparamSearch.fit(xTrain, yTrain)

    print("Optimal hyperparameters: ", hyperparamSearch.best_params_)

    optimisedRandF = hyperparamSearch.best_estimator_
    pred = optimisedRandF.predict(xTest)
    dump(optimisedRandF, 'optimisedRandF.joblib')
    
    accuracy = accuracy_score(yTest, pred)
    print(f"Accuracy = {accuracy}")
    print(classification_report(yTest, pred))
    #TODO - also figure out a way to save the cnn model so u don't have to reload it every time.
    #TODO - figure out a way to put in training data

def loadImages(directory):
    pat = f"{directory}/*.npy"
    images = []
    labels = []
    counter = 0

    fileLs = glob.glob(pat)
    for file in fileLs:
        counter += 1
        print(counter)
        image = np.load(file)
        standardiseImage = resize(image, (512,512), anti_aliasing=True)
        standardiseImage = standardiseImage/255 #Normalise the image
        images.append(standardiseImage)
        label = os.path.basename(file)
        label = label[0]
        print(label)
        labels.append(label)
        
    outImages = np.array(images)
    return outImages, labels

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

def augmentImages(image):
    pass


def dataExtraction(model):
    model.add(Input(shape=(512,512,3))) # Images are 512 by 512 and RGB
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))#Break the image into separate sub-image
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    return model


#from keras.preprocessing import ImageDataGenerator

#Data augmentation stuff
# augmentData = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

# trainGen = augmentData.flow_from_directory(
#     './data/Training/glioma_tumor',
#     target_size=(512,512),
#     batch_size=100,
#     class_mode='categorical'
# )
#Begin by augmenting the data set. Apply random scaling and rotations to do this.


main()
