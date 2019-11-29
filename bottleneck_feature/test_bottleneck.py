from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input,VGG16
import pickle

img = './floral_851505483.png'

model = load_model('./bottleneck_fc_model')
with open('./bottleneck_label.pickle') as f:
    label_ = pickle.load(f)

def predict():
    img = load_img(img, target_size=(150,150))
    # convert the image pixels to a numpy array
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # prepare the image for the VGG model
    img = preprocess_input(img)
    model = VGG16(include_top=False, weights='imagenet')
    bottleneck_feature = model.predict(img)
    res =model.predict(bottleneck_feature)
    p = label_.inverse_transform(np.argmax(res,axis=1))
    print(p[0])


if __name__ =='__main__':
    predict()
