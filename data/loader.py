''' Functions related to the handling of data.'''

import wget
import os
import tarfile
import shutil
import pandas as pd
import numpy as np

dataset_url = 'https://s3.ap-south-1.amazonaws.com/dl-assignment/data.tar.gz'
dataset_dirname = "../dataset"
dataset_filename = "data.tar.gz"
file_directory = os.path.dirname(__file__)

def load_data():
    ''' Downloads and preprocesses the dataset.'''

    if os.path.isfile(os.path.join(file_directory, dataset_dirname, dataset_filename)):
        dataset = os.path.join(file_directory, dataset_dirname, dataset_filename)
    else:
        dataset = wget.download(dataset_url, os.path.join(file_directory, dataset_dirname)) # Need to fix this path.
    if tarfile.is_tarfile(dataset):
        print("Dataset Downloaded")
        if os.path.isdir(os.path.join(file_directory, "../dataset/data")):
            shutil.rmtree(os.path.join(file_directory, "../dataset/data"))
        tar = tarfile.open(dataset)
        tar.extractall(os.path.join(file_directory, dataset_dirname))
        tar.close()
    else:
        print("Dataset could not be downloaded, please check logs.")


def impute_data(ann):
    ''' Fixes the null values in the dataset.'''
    
    # TODO: Make getting the names of columns automatic.

    ann['neck'].fillna(int(ann['neck'].mode()), inplace=True)
    ann['sleeve_length'].fillna(int(ann['sleeve_length'].mode()), inplace=True)
    ann['pattern'].fillna(int(ann['pattern'].mode()), inplace=True)
    return ann


def get_split_data():
    ''' Reads the dataset and returns IDS and labels.'''

    annotation = pd.read_csv(os.path.join(file_directory, dataset_dirname, "data", "attributes.csv"))
    annotation = impute_data(annotation)
    fileids = annotation.filename
    num_ids = fileids.shape[0]
    train_ids = fileids.iloc[:int(np.floor(0.8*num_ids))]
    valid_ids = fileids.iloc[int(np.floor(0.8*num_ids)):]
    labels = {}
    for ind in annotation.index:
        labels[annotation['filename'][ind]] = [int(annotation['neck'][ind]), int(annotation['sleeve_length'][ind]), int(annotation['pattern'][ind])]
    return (train_ids, valid_ids, labels)
