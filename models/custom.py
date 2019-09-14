'''  Multi label attribute classification simple way.'''

# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Input
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf

learning_rate = 1e-3
epochs = 10

class FlixNet:

    def __init__(self, backbone='ResNet50'):
        self.backbone = backbone
        self.neck_classes = 7
        self.sleeve_classes = 4
        self.pattern_classes = 10

    def build_base(self):
        
        base_model = ''
        if self.backbone == 'ResNet50':
            base_model = ResNet50(include_top=False)
        elif self.backbone == 'VGG19':
            base_model = VGG19(include_top=False)
        elif self.backbone == 'InceptionV3':
            base_model == InceptionV3(include_top=False)
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        return x, base_model.input

    def build_neck(self, base):

        neck = Dense(self.neck_classes)(base)
        neck = Activation('softmax', name='neck_output')(neck)
        return neck

    def build_sleeve(self, base):
        
        sleeve = Dense(self.sleeve_classes)(base)
        sleeve = Activation('softmax', name='sleeve_output')(sleeve)
        return sleeve

    def build_pattern(self, base):
        
        pattern = Dense(self.pattern_classes)(base)
        pattern = Activation('softmax', name='pattern_output')(pattern)
        return pattern

    def build():

        flix = FlixNet()
        base, base_input = flix.build_base()
        neck = flix.build_neck(base)
        sleeve = flix.build_sleeve(base)
        pattern = flix.build_pattern(base)

        model = Model(inputs=base_input, outputs=[neck, sleeve, pattern], name='FlixNet')
        return model


def get_model():
    model = FlixNet.build()
    losses = {  'neck_output': 'categorical_crossentropy',
                'sleeve_output': 'categorical_crossentropy',
                'pattern_output': 'categorical_crossentropy'}

    opt = Adam(lr=learning_rate, decay=learning_rate/epochs)
    model.compile(optimizer=opt, loss=losses, metrics=['accuracy'])
    return model




