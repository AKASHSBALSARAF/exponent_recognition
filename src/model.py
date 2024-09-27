from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Parameters
img_size = (64, 64)
num_classes_base = 101  # Assuming base numbers are from 1 to 100
num_classes_exponent = 101  # Assuming exponent values are from 1 to 100

def build_model(input_shape=(img_size[0], img_size[1], 1)):
    input_layer = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten layer
    x = Flatten()(x)

    # Base number output
    base_output = Dense(128, activation='relu')(x)
    base_output = Dense(num_classes_base, activation='softmax', name='base_output')(base_output)

    # Exponent output
    exp_output = Dense(128, activation='relu')(x)
    exp_output = Dense(num_classes_exponent, activation='softmax', name='exponent_output')(exp_output)

    # Define the model
    model = Model(inputs=input_layer, outputs=[base_output, exp_output])

    # Compile the model
    model.compile(optimizer='adam',
                  loss={'base_output': 'sparse_categorical_crossentropy',
                        'exponent_output': 'sparse_categorical_crossentropy'},
                  metrics={'base_output': 'accuracy',
                           'exponent_output': 'accuracy'})

    return model

