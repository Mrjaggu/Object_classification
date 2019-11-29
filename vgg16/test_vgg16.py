from keras.models import load_model
from keras.preprocessing import image
import pickle

model_path = './aug_vgg16_model.h5'

model = load_model(model_path)


with open('./image_class.pickle', 'rb') as f:
    classes = pickle.load(f)

def predict(query_image):
    img_path = (query_image)
    img = image.load_img(img_path, target_size=(150,150))
    img_tensor = image.img_to_array(img)
    res = model.predict(img_tensor.reshape((1,150,150,3)))
    key_list = list(classes.keys())
    result = key_list[np.argmax(res,axis=1)[0]]

    return result


if __name__ == '__main__':
    predict()
