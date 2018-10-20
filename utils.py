import os 
import sys 
import constants
import glob
from sklearn.model_selection import train_test_split
# from sympy import *
import numpy as np
import cv2 

def get_labels(folders):
    substr1 = 'subdataset/'
    interval = len(substr1)
    labels = list()
    for folder in folders:
        idx = folder.find(substr1)
        label = folder[idx+interval:len(folder)]
        labels.append(label)
    return labels

def build_fns_labels(labels, test_ratio = 0.2, val_ratio=0.2):
    train_paths = list()
    val_paths = list()
    test_paths = list()
    train_ratio = (1-val_ratio)*(1-test_ratio)
    val_ratio = (1-test_ratio)*val_ratio
    for idx, label in enumerate(labels):
        label_paths = list()
        label_dir = os.path.join(constants.DATASET_PATH, label)
        fns = glob.glob(label_dir+'/*.png')
        for fn in fns:
            label_paths.append([fn, idx])
        train_pivot = int(train_ratio*len(label_paths))
        val_pivot = train_pivot + int(val_ratio*len(label_paths))
        
        for path in label_paths[:int(train_ratio*len(label_paths))]:
            train_paths.append(path)
        for path in label_paths[train_pivot:val_pivot]:
            val_paths.append(path)
        for path in label_paths[val_pivot:]:
            test_paths.append(path)

    return train_paths, val_paths, test_paths

def split_datasets(dataset):
    fns, labels = list(zip(*dataset))
    fns = list(fns)
    idx = list(fns)
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(fns, labels, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=0.25, random_state=42, 
        shuffle=True)
    trainset = list(zip(X_train, y_train))
    validateset = list(zip(X_validation, y_validation))
    testset = list(zip(X_test, y_test))
    return trainset, validateset, testset

def spectral_local_histogram(img, window_size):
    gabor_kernel = cv2.getGaborKernel((3,3), 10 )