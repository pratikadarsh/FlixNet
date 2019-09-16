''' Functions related to the handling of data.'''

import wget
import os
import tarfile
import shutil
import pandas as pd
import numpy as np
import cv2 as cv
from data.impute import impute_data

file_directory = os.path.dirname(__file__)

def impute_data_with_most_frequent(ann_data):
     """ Fixes the null values in the dataset."""
    
    # TODO: Make getting the names of columns automatic.
    ann_data['neck'].fillna(int(ann_data['neck'].mode()), inplace=True)
    ann_data['sleeve_length'].fillna(int(ann_data['sleeve_length'].mode()), inplace=True)
    ann_data['pattern'].fillna(int(ann_data['pattern'].mode()), inplace=True)
    return ann_data

def verify_and_impute(images, ann):
    ''' Parses the list of files and verifies the corresponding image.'''

    data = pd.read_csv(ann)

    data = impute_data(data)
    drop_indices = []
    for index, row in data.iterrows():
        img = cv.imread(os.path.join(images, row['filename']))
        if img is None:
            drop_indices.append(index)
            print("File {}: {} doesn't exist. Dropping from file list".format(index, row['filename']))
    dropped =  data.drop(drop_indices).reset_index().drop('index', axis=1)
    os.remove(ann)
    dropped.to_csv(ann, index=False)
    return ann
            

def get_split_data(annotation_path, train_split):
    ''' Reads the dataset and returns IDS and labels.'''

    annotation = pd.read_csv(annotation_path)
    fileids = annotation.filename.tolist()
    train_ids = fileids[:int(train_split*len(fileids))]
    valid_ids = fileids[int(train_split*len(fileids)):]
    labels = {}
    for ind in annotation.index:
        labels[annotation['filename'][ind]] = [int(annotation['neck'][ind]),
                int(annotation['sleeve_length'][ind]), int(annotation['pattern'][ind])]
    return (train_ids, valid_ids, labels)
