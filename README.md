# FlixNet - Multi Label Fashion Attribute Classification

This is an assignment solution for a multi label attribute classification problem. Using this repo, you can train and test a model from two choices:

 - Custom CNN 
 - Bilinear CNN

This codebase performs the following activites and can be used as a reference for creating a pipeline for your computer vision based tasks:

 - **Data pipeline** using generators.
 - **Data augmentation** using image-processing (check **augment.py** in data folder).
 - **Data imutation** to handle null values in the annotations.
 - **Multi-Label Model** definition and creation using keras.


# Getting Started

For an overview of how to load a dataset, augment it , train a model and finally perform inference using the trained weights, check out **flixNet_demo.ipynb**. This jupyter notebook makes use of all the scripts present in the repo to present a clear picture of its functioning. 

# Prerequisites

This codebase mainly requires keras and Tensorflow to function. For a detailed list of required packages, check ***requirements.txt***. Note that this repo doesn't need installation on a system, you can directly use the scripts provided.

# Training

The training is performed using the **flixNet.py** file. Training script requires two mandatory arguments i.e path to images and path to annotation csv file. As an output of training, trained weights and training logs get saved in the **logs** folder. After you have downloaded and placed your dataset in a folder, you can start training using the following command:

    python3 flixNet.py --mode train --images <path_to_images> --csv <path_to_annotation> 

# Inference

Similar to training, inference is also performed using the **flixNet.py** file. In this case, the script requires just one argument i.e the path to the images. As an output of the inference, an **output.csv** file gets saved. For running inference using **flixNet.py**, run the following command:

    python3 flixNet.py --mode test --images <path_to_images>

# Further Help

    usage: flixNet.py [-h] [--mode MODE] --images IMAGES [--csv CSV]
                      [--log_dir LOG_DIR] [--saved_weights SAVED_WEIGHTS]
                      [--output_weights OUTPUT_WEIGHTS] [--backbone BACKBONE]
                      [--arch ARCH] [--batch_size BATCH_SIZE] [--lr LR]
                      [--epochs EPOCHS] [--train_split TRAIN_SPLIT]
    
    optional arguments:
      -h, --help            show this help message and exit
      --mode MODE           [train, test]
      --images IMAGES       Path to folder containing images for either training
                            or testing.
      --csv CSV             Path to the annotation csv file.
      --log_dir LOG_DIR     Path to save training logs.
      --saved_weights SAVED_WEIGHTS
                            Path to saved weight file to be used for inference.
      --output_weights OUTPUT_WEIGHTS
                            Path to save weight file after training.
      --backbone BACKBONE   Base CNN-> [ResNet50,VGG19,InceptionV3]
      --arch ARCH           Deep Learning model architecture. (Bilinear CNN or
                            custom)
      --batch_size BATCH_SIZE
                            Number of images to train at once in a single step.
      --lr LR               Initial learning rate.
      --epochs EPOCHS       Number of epochs for training.
      --train_split TRAIN_SPLIT
                            The percentage of training data.

# References

 - https://www.groundai.com/project/improving-the-annotation-of-deepfashion-images-for-fine-grained-attribute-recognition/
 - http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf
 - http://cs231n.stanford.edu/reports/2015/pdfs/BLAO_KJAG_CS231N_FinalPaperFashionClassification.pdf
 - http://cs231n.stanford.edu/reports/2016/pdfs/286_Report.pdf
 - http://chenlab.ece.cornell.edu/people/Andy/publications/ECCV2012_ClothingAttributes.pdf
 - https://pdfs.semanticscholar.org/ce4a/d1ad4134d9131af21d4213e598f03475cfd3.pdf
 - http://myweb.sabanciuniv.edu/berrin/files/2018/12/icme-final-paper.pdf
 - https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779

# TODO

 - Running training using bilinear cnn causes a memory crash due to large ram usage. This might be solved by changing the output tensors.
 - Write logger.
 - Data imutation using conditional probability.
 - Add training strategy of training with backbone frozen and then unfrozen.

