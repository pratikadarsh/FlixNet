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
from keras.layers.core import Reshape
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Input
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras import backend as k
import tensorflow as tf

learning_rate = 1e-3
epochs = 10

class FlixNet:

    def __init__(self, backbone='ResNet50'):
        self.backbone = backbone
        self.height = 224
        self.width = 224
        self.neck_classes = 7
        self.sleeve_classes = 4
        self.pattern_classes = 10
        self.no_last_layer = -1

    def _outer_product(self, x):
        """ Calculate outer-products of two tensors. """
        return k.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]

    def _signed_sqrt(self, x):
        """ Calculate element-wise signed square-root. """
        return k.sign(x) * k.sqrt(k.abs(x) + 1e-9)

    def _l2_normalize(self, x, axis=-1):
        """Calculate L2 normalization."""
        return k.l2_normalize(x, axis=axis)

    def build_base(self):
        
        input_tensor = Input(shape=[self.height, self.width, 3])
        base_model = ''
        if self.backbone == 'ResNet50':
            base_model = ResNet50(input_tensor=input_tensor, include_top=False)
        elif self.backbone == 'VGG19':
            base_model = VGG19(input_tensor=input_tensor, include_top=False)
            self.no_last_layer = 17
        elif self.backbone == 'InceptionV3':
            base_model == InceptionV3(input_tensor=input_tensor, include_top=False)
        return base_model

    def build_fc(self, base):

        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation = 'relu')(x)
        x = Dense(512, activation='relu')(x)
        return x

    def build_neck(self, cnn):

        neck = Dense(self.neck_classes)(cnn)
        neck = Activation('softmax', name='neck_output')(neck)
        return neck

    def build_sleeve(self, cnn):
        
        sleeve = Dense(self.sleeve_classes)(cnn)
        sleeve = Activation('softmax', name='sleeve_output')(sleeve)
        return sleeve

    def build_pattern(self, cnn):
        
        pattern = Dense(self.pattern_classes)(cnn)
        pattern = Activation('softmax', name='pattern_output')(pattern)
        return pattern

    def build_bcnn(self, base):
        """ Create bilinear CNN architecture. """

        # Feature extraction from detector
        model_detector = base
        output_detector = model_detector.layers[self.no_last_layer].output
        shape_detector = model_detector.layers[self.no_last_layer].output_shape

        # Extract features from extractor
        model_extractor = base
        output_extractor = model_extractor.layers[self.no_last_layer].output
        shape_extractor = model_extractor.layers[self.no_last_layer].output_shape
        print(shape_extractor)

        # Reshape tensor to (minibatch_size, total_pixels, filter_size)
        output_detector = Reshape(
            [shape_detector[1]*shape_detector[2], shape_detector[-1]])(output_detector)
        output_extractor = Reshape(
            [shape_extractor[1]*shape_extractor[2], shape_extractor[-1]])(output_extractor)

        # Outer-products
        x = Lambda(self._outer_product)([output_detector, output_extractor])
        # Reshape tensor to (minibatch_size, filter_size_detector*filter_size_extractor)
        x = Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
        # Signed square-root
        x = Lambda(self._signed_sqrt)(x)
        # L2 normalization
        x = Lambda(self._l2_normalize)(x)            
        
        # Add final FC layers.
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        return x

    def build(arch='bcnn'):

        flix = FlixNet()
        base = flix.build_base()

        if arch == 'custom':
            cnn = flix.build_fc(base)
        elif arch == 'bcnn':
            cnn = flix.build_bcnn(base)
        neck = flix.build_neck(cnn)
        sleeve = flix.build_sleeve(cnn)
        pattern = flix.build_pattern(cnn)
        model = Model(inputs=base.input, outputs=[neck, sleeve, pattern], name='FlixNet')
        return model

def get_model(arch):
    model = FlixNet.build(arch)
    losses = {  'neck_output': 'categorical_crossentropy',
                'sleeve_output': 'categorical_crossentropy',
                'pattern_output': 'categorical_crossentropy'}

    opt = Adam(lr=learning_rate, decay=learning_rate/epochs)
    model.compile(optimizer=opt, loss=losses, metrics=['accuracy'])
    return model




