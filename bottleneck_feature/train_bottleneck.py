import numpy as np
import keras
import keras.backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,LeakyReLU,BatchNormalization


x_train=np.load('/content/drive/My Drive/GreenDeck/15k_data_cnn_features.npy')
y_train=np.load('/content/drive/My Drive/GreenDeck/15k_data_cnn_feature_asins.npy')




class Custom_lr(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
	    K.set_value(self.model.optimizer.lr, 0.001)

    def on_epoch_begin(self, epoch, logs={}):
        lr_present=K.get_value(self.model.optimizer.lr)
        #print(epoch)
        if (epoch%10==0) and epoch:

            K.set_value(self.model.optimizer.lr, lr_present/((epoch)**0.5))
            print(K.get_value(self.model.optimizer.lr))
            print(lr_present/((epoch)**0.5))

top_model=Sequential()
top_model.add(Flatten(input_shape=xtrain.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(BatchNormalization())
top_model.add(Dropout(0.5))
top_model.add(Dense(32, activation='relu'))
top_model.add(BatchNormalization())
top_model.add(Dropout(0.5))
# top_model.add(BatchNormalization())
top_model.add(Dense(17, activation='softmax'))

top_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
	    Custom_lr()
	]
checkpoint = ModelCheckpoint("./bottleneck_vgg16_model.h5",
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



top_model.fit(xtrain, ytrain,
          epochs=30,
          batch_size=32,
          validation_data=(xtest, ytest), callbacks=callbacks)

score = top_model.evaluate(xtest,ytest)
print(score)
