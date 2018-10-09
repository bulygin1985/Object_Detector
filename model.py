from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Reshape, Activation, BatchNormalization

def get_model(input_w, input_h, GRID_W, GRID_H):

    img_input = Input(shape=(input_w, input_h, 3), name = 'image_input')

    x = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x) 

    x = Conv2D(filters = 24, kernel_size = (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x) 

    x = Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x) 

    x = Conv2D(filters = 48, kernel_size = (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x) 

    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x) 

    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(x) #experiment layer
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters = 5, kernel_size = (3, 3), padding = 'same')(x)

    # instead of last conv layer one may use fully-connected head. Model size is inreased from 800 Kb to 25 Mb
    # accuracy is slightly decreased, but model is trained faster
    
    # x = Flatten(name='flatten')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(5*GRID_W*GRID_H)(x)
    # x = Reshape((GRID_W, GRID_H, 5))(x)

    model = Model(inputs=img_input, outputs=x)
    return model