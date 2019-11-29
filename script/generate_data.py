#data generation
from keras.preprocessing.image import ImageDataGenerator


#data path---
folder = './GreenDeck/train_test/'

def data_gen():
    train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
    valid_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
    test_datagen = ImageDataGenerator()

    training_set = train_datagen.flow_from_directory(folder + 'train/',target_size=(150,150),batch_size=8,class_mode = 'categorical')
    valid_set = valid_datagen.flow_from_directory(folder + 'val/',target_size=(150,150),batch_size=8,class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory(folder + 'test/',target_size=(150,150),batch_size=8,class_mode = None,)

    return training_set,valid_set,test_set
