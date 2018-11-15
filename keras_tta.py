import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.
    Args:
    model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model
        self.gene = datagen = ImageDataGenerator(
           rotation_range=180,
           width_shift_range=0.1,
           height_shift_range=0.1,
           shear_range=20,
           zoom_range=[0.8, 1.2],
           fill_mode='reflect',
           horizontal_flip=True,
           vertical_flip=True)

    def predict_tta(self, X, aug_times=16):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.
        Args:
            X (numpy array of dim 4): The data to get predictions for.
        """

        pred = []
        for x_i in X:
            sum_p = 0
            for i, d in enumerate(self.gene.flow(x_i[np.newaxis], batch_size=1)):
                if i >= aug_times:
                    break
                p = self.model.predict(d)[0]
                sum_p += p
            pred.append(sum_p/aug_times)
        return np.array(pred)

