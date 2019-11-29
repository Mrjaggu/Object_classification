from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D,Dense,Dropout
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras import optimizers
from generate_data import *


training_set,valid_set,test_set = data_gen()

batch_size=8
training_set.target_size,valid_set.target_size,test_set.target_size
train_size = training_set.n
valid_size = valid_set.n
test_size =test_set.n
train_size,valid_size,test_size




def train_model():
    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(17, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # train the model on the new data for a few epochs

    for layer in model.layers[:25]:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=optimizers.SGD(lr=1e-4,momentum=0.99), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("./vgg16_model.h5",
                                monitor="val_loss",
                                mode="min",
                                save_best_only = True,
                                verbose=1)
    earlystop = EarlyStopping(monitor = 'val_loss',
                              mode="min",
                              min_delta = 0,
                              patience = 5,
                              verbose = 1,
                              restore_best_weights = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2,verbose = 1, min_delta = 0.0001)

    #combining in single so that we will pass callbacks in model.fit
    callbacks = [checkpoint,earlystop,reduce_lr]

    #train the model
    history = model.fit_generator(training_set,steps_per_epoch =4000,epochs = 12,validation_data = valid_set,validation_steps  = 100,callbacks=callbacks)

    #evaluating the model
    eval = model.evaluate_generator(generator=valid_set,steps=valid_size/batch_size)
    print('accuracy=',eval[1])

if __name__ == '__main__':
    train_model()
