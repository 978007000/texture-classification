import os 
import sys 
import constants
import glob

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
