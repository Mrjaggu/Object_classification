import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from PIL import Image
import pandas as pd
import os


# dimensions of our images.
img_width, img_height = 150,150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = './images_folder/'
nb_train_samples = len(os.listdir(train_data_dir))
epochs = 50
batch_size = 1

#function to create cnn features for image...
def save_bottlebeck_features():
    asins = []
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    for i in generator.filenames:
        asins.append(i[2:-5])

    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    bottleneck_features_train = bottleneck_features_train.reshape((nb_train_samples,bottleneck_features_train.shape[1]*bottleneck_features_train.shape[2]*bottleneck_features_train.shape[3]))

    np.save(open('./15k_data_cnn_features.npy', 'wb'), bottleneck_features_train)
    np.save(open('./15k_data_cnn_feature_asins.npy', 'wb'), np.array(asins))


save_bottlebeck_features()
