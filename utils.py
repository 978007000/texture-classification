import os 
import sys 
import constants
import glob
from sklearn.model_selection import train_test_split
from sympy import *
import numpy as np

def get_labels(folders):
    substr1 = 'subdataset/'
    interval = len(substr1)
    labels = list()
    for folder in folders:
        idx = folder.find(substr1)
        label = folder[idx+interval:len(folder)]
        labels.append(label)
    return labels

def build_fns_labels(labels):
    dataset = list()
    for idx, label in enumerate(labels):
        label_dir = os.path.join(SUBDATASET_DIR, label)
        fns = glob.glob(label_dir+'/*.png')
        print(label)
        print(len(fns))
        for fn in fns:
            dataset.append([fn, idx])
    return dataset

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

def spectral_local_histogram(img, window_size)