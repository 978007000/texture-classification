import os 
import sys 
import constants
import glob
from random import shuffle

SUBDATASET_DIR = os.path.join(constants.DATA_DIR, 'subdataset')

folders = [x[0] for x in os.walk(SUBDATASET_DIR)]

print(folders)
del folders[0]

def get_labels(folders):
    substr1 = 'subdataset/'
    interval = len(substr1)
    labels = list()
    for folder in folders:
        idx = folder.find(substr1)
        label = folder[idx+interval:len(folder)]
        labels.append(label)
    return labels

labels = get_labels(folders)
print(labels)

def build_images_path_label(labels):
    dataset = list()
    for idx, label in enumerate(labels):
        label_dir = os.path.join(SUBDATASET_DIR, label)
        fns = glob.glob(label_dir+'/*.png')
        print(label)
        print(len(fns))
        for fn in fns:
            dataset.append([fn, idx])
    return dataset


dataset = build_images_path_label(labels)
shuffle(dataset)
print(dataset)