from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.applications import Xception


class MyXception:
    def create_model(input_shape, n_out, without_sigmoid=False):
        input_tensor = Input(shape=input_shape)
        base_model = Xception(include_top=False,
                        weights='imagenet',
                        pooling='max')
        x = base_model(input_tensor)
        if without_sigmoid == False:
            output = Dense(n_out, activation='sigmoid')(x)
        else:
            output = Dense(n_out)(x)
            print('withoud sigmoid')
        model = Model([input_tensor], [output])
        model.name = 'xception'

        return model
