from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model

        self.model = Sequential([
            
            # Rescaling normalizes pixels to between 1 and 255.
            layers.Rescaling(1./255, input_shape=input_shape),
            # Resizes images.
            layers.Resizing(32,32),
            layers.Conv2D(4, 2, padding='same', activation='relu'),      
            layers.MaxPooling2D(),
            # When we go from convolutional layers to fully connected layers.       
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])
        pass
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        optimizer = RMSprop(learning_rate=0.001)  # You can choose your optimizer here
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        pass