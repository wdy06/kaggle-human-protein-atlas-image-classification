from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3


class MyInceptionV3:
    def create_model(input_shape, n_out):
        input_tensor = Input(shape=input_shape)
        base_model = InceptionV3(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)
        bn = BatchNormalization()(input_tensor)
        x = base_model(bn)
        x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(n_out, activation='sigmoid')(x)
        model = Model(input_tensor, output)
        model.name = 'inceptionV3'

        return model
