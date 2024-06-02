#%%
import os
import numpy as np
from keras import layers
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.backend import clear_session

import os
import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array

#%%
batch_size = 64
epochs = 10
lr = 1e-4

script_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(script_dir, '..', 'binary_data', 'train_images')
train_dir = os.path.normpath(train_dir)  # Normalize the path

test_dir = os.path.join(script_dir, '..', 'binary_data', 'test_images')
test_dir = os.path.normpath(test_dir)  # Normalize the path


#%%
class CustomDataGenerator(Sequence):
    def __init__(self, directory, batch_size=32, target_size=(256, 256, 3), shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.classes = sorted(os.listdir(directory))
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.image_paths = []
        for cls in self.classes:
            cls_path = os.path.join(directory, cls)
            self.image_paths += [(os.path.join(cls_path, file), cls) for file in os.listdir(cls_path)]
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]

        X = []
        y = []
        for path, label in batch_paths:
            img = load_img(path, target_size=self.target_size)
            img_array = img_to_array(img)
            X.append(img_array)
            y.append(self.class_indices[label])

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

#%%
def create_model(image_size=(256, 256, 3), initial_filters=64, num_blocks=3, dropout_rate=0.25):
    # Define the input layer
    inputs = tf.keras.Input(shape=image_size)

    x = layers.Rescaling(1.0 / 255)(inputs)
    
    # First block
    x = layers.Conv2D(initial_filters, 3, strides=2, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Inner blocks with dropout and increasing filter sizes
    for i in range(1, num_blocks + 1):
        x = layers.Conv2D(initial_filters * (2 ** i), 3, strides=2, padding="same", activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Flatten the output and add the final dense layer
    x = layers.Flatten()(x)
    x = layers.Dense(initial_filters, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# List of different image sizes and number of inner layers
# image_sizes = [128, 256, 512, 1024]
image_sizes = [256]
layer_sizes = [0, 1, 2, 3]

histories = []
for image_size in image_sizes:
    image_res = (image_size, image_size, 3)
    if image_size in [128,256]:
        train_generator = CustomDataGenerator(directory=train_dir, batch_size=batch_size*2, target_size=image_res)
        validation_generator = CustomDataGenerator(directory=test_dir, batch_size=batch_size*2, target_size=image_res)
#%%
    else:
        train_generator = CustomDataGenerator(directory=train_dir, batch_size=batch_size, target_size=image_res)
        validation_generator = CustomDataGenerator(directory=test_dir, batch_size=batch_size, target_size=image_res)
    for num_layers in layer_sizes:
        try:
            clear_session()
            print(f"Creating and training model with image size {image_size}x{image_size} and {num_layers} inner layers")
            model = create_model(image_size=image_res, num_blocks=num_layers)
            
            # Compile the model
            optimizer = Adam(learning_rate=lr)
            model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

            model.summary()
            # Train the model and store the history
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator
            )
            
            # Store the history with the configuration
            histories.append((image_size, num_layers, history))
            print("\n" + "="*50 + "\n")
        except:
            print("\n" + "="*50 + "\n")
            print('Model training issue')
            print("\n" + "="*50 + "\n")
#%%
def plot_history(history, title, save_path):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    
    # Show the plot
    plt.show()

# Visualize the results for each combination and save the plots
for image_size, num_layers, history in histories:
    title = f'Image Size: {image_size}, Layers: {num_layers}'
    save_path = f'./graphs/{image_size}_{num_layers}.png'
    plot_history(history, title, save_path)