from tensorflow import keras

def cnn_model(input_shape=(28, 28, 1), num_classes=3):
    """it builds the cnn block

    Args:
        input_shape (tuple, optional): the input shape of imgs. Defaults to (28, 28, 1).
        num_classes (int, optional): how many classes. Defaults to 3.

    Returns: model
    """    
    model = keras.models.Sequential()

    # Specify the input shape
    model.add(keras.Input(shape=input_shape))

    # 1st convolutional block
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))

    # 2nd convolutional block
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))

    # 3rd convolutional block
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model