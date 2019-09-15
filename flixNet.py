import argparse
import keras
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from data.datagen import *
from data.loader import *
from models.model import *


def train(args):
    """ Trains and saves the model."""

    #Verify the annotation file.
    ann_file_path = verify_and_impute(args.images, args.csv)
    train, valid, labels = get_split_data(args.csv, args.train_split)

    training_generator = DataGenerator(train, args.images, labels, args.batch_size)
    validation_generator = DataGenerator(valid, args.images, labels, args.batch_size)

    model = get_model(args.arch)
    #Callbacks.
    logging = TensorBoard(log_dir=args.log_dir)
    checkpoint = ModelCheckpoint(os.path.join(args.log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                            monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=args.epochs,
                        use_multiprocessing=True,
                        workers=5,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    model.save(args.output_weights)

def inference(args):
    """ Perform inference using a trained model."""

    model = load_model(args.saved_weights)
    model.summary()

def init_args():
    """ Reads command line arguments."""
    
    #Construct argument parser.
    parser = argparse.ArgumentParser()

    #General settings.
    parser.add_argument('--mode', type=str, default='test',
                        help="[train, test]")
    parser.add_argument('--images', type=str, required=True,
                        help="Path to folder containing images for either training or testing.")
    parser.add_argument('--csv', type=str, default='',
                        help="Path to the annotation csv file.")
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help="Path to save training logs.")
    parser.add_argument('--saved_weights', type=str, default='./models/flixNet.h5',
                        help="Path to saved weight file to be used for inference.")
    parser.add_argument('--output_weights', type=str, default='./models/flixNet.h5',
                        help="Path to save weight file after training.")
    parser.add_argument('--backbone', type=str, default='ResNet50',
                        help="Base CNN-> [ResNet50,VGG19,InceptionV3]")
    parser.add_argument('--arch', type=str, default='custom',
                        help="Deep Learning model architecture. (Bilinear CNN or custom)")
    
    #Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of images to train at once in a single step.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Number of epochs for training.")
    parser.add_argument('--train_split', type=float, default= 0.9,
                        help="The percentage of training data.")

    return parser.parse_args()

def main():

    args = init_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        inference(args)
    else:
        print("Incorrect input for '--mode'. Please see help.")

if __name__ == '__main__':
    main()
