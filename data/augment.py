''' Data augmentation. '''

import os
import cv2 as cv
import numpy as np

def crop_at_center(img, size=(224,224)):
    """ Crops the image at center."""

    target_h = size[0]
    target_w = size[1]

    img_size = img.shape
    img_h = img_size[0]
    img_w = img_size[1]

    h_start = int((img_h - target_h)/2)
    w_start = int((img_w - target_h)/2)
    img_cropped = img[h_start:h_start+target_h, w_start:w_start+target_w, :]
    return img_cropped

def crop_at_random(img, size=(224,224)):
    """ Crops the image at random position."""

    target_h = size[0]
    target_w = size[1]

    img_size = img.shape
    img_h = img_size[0]
    img_w = img_size[1]

    margin_h = (img_h - target_h)
    margin_w = (img_w - target_w)
    h_start = 0
    w_start = 0
    if margin_h != 0:
        h_start = np.random.randint(low=0, high=margin_h)
    if margin_w != 0:
        w_start = np.random.randint(low=0, high=margin_w)
    img_cropped = img[h_start:h_start+target_h, w_start:w_start+target_w, :]
    return img_cropped

def horizontal_flip(img):
    """ Create a mirror image along the horizontal axis."""

    if np.random.random() >= 0.5:
        return img[:, ::-1, :]
    else:
        return img

def normalize_image(img, mean=(0., 0., 0.), std=(1.0, 1.0, 1.0)) :
    """ Feature-wise normalization. """

    img = np.asarray(img, dtype=np.float32)
    for dim in range(3):
        img[:,:,dim] = ( img[:,:,dim] - mean[dim] )/std[dim]
    return img

def preprocess(img_path):
    """ Performs all the above operations and returns list of newly created images."""

    img_list = []
    if os.path.isfile(img_path) == False:
        print("File {} doesn't exist.".format(img_path))
        return img_list
    img = cv.imread(img_path)
    center_cropped = crop_at_center(img)
    random_cropped = crop_at_random(img)
    h_flipped = horizontal_flip(img)
    normalized = normalize_image(img)
    
    img_dir = os.path.dirname(img_path)
    file_id = os.path.basename(img_path).split(".")[0]
    
    ccropped_path = os.path.join(img_dir, file_id + "_center_cropped" + ".jpg")
    if (cv.imwrite(ccropped_path, center_cropped)):
        img_list.append(ccropped_path)
    rcropped_path = os.path.join(img_dir, file_id + "_random_cropped" + ".jpg")
    if (cv.imwrite(rcropped_path, random_cropped)):
        img_list.append(rcropped_path)
    hflipped_path = os.path.join(img_dir, file_id + "_horizontal_flipped" + ".jpg")
    if (cv.imwrite(hflipped_path, h_flipped)):
        img_list.append(hflipped_path)
    norm_path = os.path.join(img_dir, file_id + "_normalized" + ".jpg")
    if (cv.imwrite(norm_path, normalized)):
        img_list.append(norm_path)

    return img_list

