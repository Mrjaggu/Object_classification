from utils import *
from PIL import Image
from keras.preprocessing import image


image_numpy = []
image_label = []

source = '/home/ajay/GreenDeck/new_image'
folder = os.listdir(source)
for image_folder in tqdm(folder):
    for id,images in tqdm(enumerate(os.listdir(source + '/' + image_folder))):
        if id <=500:
            image_name = images.split('.')[0]
            image_path = source + '/' + image_folder + '/' + images
            img = image.load_img(image_path, target_size=(150,150))
            x = image.img_to_array(img)
            image_numpy.append(x)
            image_label.append(image_name)
        else:
            pass

np.save('./image_15k_feature.npy',image_numpy)
np.save('./image_15k_label.npy',image_label)
# images = np.load('/home/ajay/GreenDeck/api/image_temp.npy')
# print(len(images))
