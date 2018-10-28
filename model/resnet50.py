from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.applications.resnet50 import ResNet50


class MyResNet50:
    def create_model(input_shape, n_out):
        input_tensor = Input(shape=input_shape)
        base_model = ResNet50(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)
        x = base_model(input_tensor)
        x = Flatten()(x)
        output = Dense(n_out, activation='sigmoid')(x)
        model = Model(input_tensor, output)
        model.name = 'resnet50'

        return model
