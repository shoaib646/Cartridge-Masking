import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from random import randint
from PIL import Image
from datetime import datetime
import random
import json
import os





# Constants (default)
DATA_SAVE_PATH = 'processed_data.npz'
MODEL_SAVE_PATH = 'model_for_cartridge.h5'
TRAIN_PATH = './Stage1_train/'
TEST_PATH = "./Stage1_test/"
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3


class ImagePreprocessor:
    def __init__(self, train_path, test_path, img_height, img_width, channels=3):
        self.train_path = train_path
        self.test_path = test_path
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels

    def _load_and_resize_image(self, img_path):
        grayscale_image = imread(img_path)
        expanded_image = np.expand_dims(grayscale_image, axis=-1)
        img = np.repeat(expanded_image, 3, axis=-1)
        img = resize(img, (self.img_height, self.img_width), mode='constant', preserve_range=True)
        return img

    def _load_and_process_mask(self, mask_path):
        mask_ = imread(mask_path)
        mask_ = np.expand_dims(resize(mask_, (self.img_height, self.img_width), mode='constant', preserve_range=True), axis=-1)
        return mask_

    def _process_data(self, ids, data_path, is_train=True):
        data = np.zeros((len(ids), self.img_height, self.img_width, self.channels), dtype=np.uint8)
        labels = np.zeros((len(ids), self.img_height, self.img_width, 1), dtype=bool)

        print(f"Resizing {'training' if is_train else 'test'} masks and/or only images.")
        for i, id_ in tqdm(enumerate(ids), total=len(ids)):
            path = os.path.join(data_path, id_)
            img_path = os.path.join(path, 'images', f'{id_}.png')

            img = self._load_and_resize_image(img_path)
            data[i] = img

            if is_train:
                mask_path = os.path.join(path, 'masks')
                mask = np.zeros((self.img_height, self.img_width, 1), dtype=bool)
                for mask_file in os.listdir(mask_path):
                    mask_file_path = os.path.join(mask_path, mask_file)
                    mask_ = self._load_and_process_mask(mask_file_path)
                    mask = np.maximum(mask, mask_)
                labels[i] = mask

        return data, labels

    def process_train_data(self):
        train_ids = next(os.walk(self.train_path))[1]
        X_train, y_train = self._process_data(train_ids, self.train_path, is_train=True)
        return X_train, y_train

    def process_test_data(self):
        test_ids = next(os.walk(self.test_path))[1]
        X_test, sizes_test = self._process_data(test_ids, self.test_path, is_train=False)
        return X_test, sizes_test



if os.path.isfile(DATA_SAVE_PATH):
    print(f"Loading processed data from {DATA_SAVE_PATH}")
    loaded_data = np.load(DATA_SAVE_PATH)
    X_train = loaded_data['X_train']
    y_train = loaded_data['y_train']
    X_test = loaded_data['X_test']
    sizes_test = loaded_data['sizes_test']
else:
    # Process data
    image_preprocessor = ImagePreprocessor(TRAIN_PATH, TEST_PATH, IMG_HEIGHT, IMG_WIDTH)
    X_train, y_train = image_preprocessor.process_train_data()
    X_test, sizes_test = image_preprocessor.process_test_data()

    # Save processed data
    np.savez(DATA_SAVE_PATH, X_train=X_train, y_train=y_train, X_test=X_test, sizes_test=sizes_test)
    print('Processed data saved to', DATA_SAVE_PATH)


def unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)):
    inputs = Input(input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Encoder
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    conv1 = tf.keras.layers.Dropout(0.1)(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = tf.keras.layers.Dropout(0.1)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = tf.keras.layers.Dropout(0.3)(conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

    # Decoder
    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = tf.keras.layers.Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = tf.keras.layers.Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)

    up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = tf.keras.layers.Dropout(0.1)(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)

    up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = tf.keras.layers.Dropout(0.1)(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    # Compiling the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model




callbacks = [
    ModelCheckpoint('model_for_cartridge.h5', verbose=1, save_best_only=True),
    EarlyStopping(patience=5, monitor='val_loss'),
    TensorBoard(log_dir='logs')
]

if os.path.isfile(MODEL_SAVE_PATH):
    print(f"Loading model from {MODEL_SAVE_PATH}")
    unet_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    with open('training_history.json', 'r') as json_file:
        history = json.load(json_file)

        # Assuming you have stored the training history during training
        # if 'history' in locals():
            # Visualizing training history
        plt.figure(num="Model Performance History")  # Set the window name
        plt.subplot(2, 1, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        
else:
    # Train model
    unet_model = unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    unet_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Model training
    history = unet_model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=16, callbacks=callbacks)

    history_dict = history.history
    with open('training_history.json', 'w') as json_file:
        json.dump(history_dict, json_file)

    # Save trained model
    unet_model.save(MODEL_SAVE_PATH)
    print('Trained model saved to', MODEL_SAVE_PATH)

# Visualizing training history
    plt.figure(num="Performance window") 
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Model predictions
preds_train = unet_model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = unet_model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = unet_model.predict(X_test, verbose=1)


idx = random.randint(0, len(X_train))
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

def display_images_with_subplots(original, ground_truth, predicted, title_original='Original', title_manual='Manual Mask', title_auto='Auto Masked', figsize=(15, 5), file_name=None, window=None):
    for i in range(3):
        ix = random.randint(0, len(predicted))
    
        try:
            fig, axes = plt.subplots(1, 3, figsize=figsize,num=window)
            
        
            axes[0].imshow(original[ix], cmap='gray')
            axes[0].set_title(title_original)
        
            axes[1].imshow(np.squeeze(ground_truth[ix]),cmap='gray')
            axes[1].set_title(title_manual)
        
            axes[2].imshow(np.squeeze(predicted[ix]),cmap='gray')
            axes[2].set_title(title_auto)
        
            if file_name:
                plt.savefig(file_name)
            plt.show()
            break
            
        except:
            continue
        

# Displaying random training image results
file_name = f'saved_plots/Training_{datetime.now().strftime("%H")}.png'
display_images_with_subplots(X_train, y_train, preds_train_t, title_original='Original (Training)', file_name=file_name, window='Trained Data')

# Displaying random validation image results
file_name = f'saved_plots/Validation_{datetime.now().strftime("%H")}.png'
display_images_with_subplots(X_train[int(X_train.shape[0]*0.9):], y_train[int(y_train.shape[0]*0.9):], preds_val_t, title_original='Original (Validation)', file_name=file_name, window='Test Data')
