import os
import random

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.layers import Lambda, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

import behavioral_cloning_utils as utils


def build_model():
    """
    Build a CNN model using Keras
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=utils.INPUT_SHAPE))  # pixel values between -1 and 1
    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))  # output layer
    model.summary()

    return model


def build_efficient_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=utils.INPUT_SHAPE))  # pixel values between -1 and 1
    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))  # output layer
    model.summary()

    return model


def load_data(data_dir):
    """
    Load all recorded images together with the recorded steering angles.
    """
    imgs = []
    steers = []

    files = os.listdir(data_dir)
    random.shuffle(files)

    print(f"Loading images... ({len(files)} total files in directory)")

    for file in files:
        if not file.endswith(".png"):
            continue

        img, steer = utils.load_image(data_dir, file, include_left_right=True, include_augmented=True)

        if img is None:
            continue

        imgs.append(utils.preprocess(img))
        steers.append(steer)

        if len(imgs) % 1000 == 0:
            print(f"{len(imgs)} labelled images loaded.")

    print(f"Loading complete! ({len(imgs)} labelled images)")
    return np.array(imgs), np.array(steers)


def train_model(data_dir, epochs=64, batch_size=64, model_name='behavioral_cloning', epoch_loss=None):
    """
    Train the CNN model for behavioral cloning.
    """
    model = build_model()
    (imgs, steers) = load_data(data_dir)

    checkpoint = ModelCheckpoint('ckpt/model-{epoch:03d}.keras',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1.0e-4))

    for i in range(epochs):
        model.fit(imgs, steers, epochs=1, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint])
        print(f'EPOCH {i}', model.history.history)

    # history = model.fit(imgs, steers, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint])
    model.save(filepath=f"models/{model_name}.keras", overwrite=True)

    return model.history


if __name__ == '__main__':
    train_model(r'C:\Users\A42893\Documents\FE\Workshop\test\track_4', epochs=16)
