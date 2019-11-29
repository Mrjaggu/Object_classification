from utils import *
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import pairwise_distances


class images_similarity():

    def __init__(self,data,image_feature,label):
        self.dress = data
        self.image_feature = image_feature
        self.label = label

    #function to test random image and return similar images..
    def new_image_similar_product(self,img, num_results):
        img = load_img(img, target_size=(150,150))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        model = VGG16(include_top=False, weights='imagenet')
        bottleneck_feature = model.predict(img)
        pairwise_dist = pairwise_distances(self.image_feature,bottleneck_feature.reshape(1,-1))
        indices = np.argsort(pairwise_dist.flatten())[0:num_results]
        pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
        uid = [int(self.label[indices[i]].split('/')[1].split('.')[0].split('_')[1]) for i in range(len(indices))]

        for i in uid:
            rows = self.dress[['image_url','category']].loc[self.dress['_unit_id']==i]
            for indx, row in rows.iterrows():
                resp = urllib.request.urlopen(row['image_url'])
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                cv2.imshow(row['category'],image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


    # function to return similar images from image...
    def get_similar_products_cnn(self,doc_id, num_results):
      unit_id = self.label[doc_id].split('/')[1].split('_')[1]
      print(self.image_feature[doc_id].shape)
      pairwise_dist = pairwise_distances(self.image_feature,self.image_feature[doc_id].reshape(1,-1))
      indices = np.argsort(pairwise_dist.flatten())[0:num_results]
      pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
      uid = [int(self.label[indices[i]].split('/')[1].split('.')[0].split('_')[1]) for i in range(len(indices))]

      for i in uid[1:]:
        rows = self.dress[['image_url','category']].loc[self.dress['_unit_id']==i]
        for indx, row in rows.iterrows():
            resp = urllib.request.urlopen(row['image_url'])
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imshow(row['category'],image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    dress = pd.read_csv('/home/ajay/GreenDeck/Data/dress_patterns.csv')
    bottleneck_features_train = np.load('/home/ajay/GreenDeck/Data/15k_data_cnn_features.npy')
    label = np.load('/home/ajay/GreenDeck/Data/15k_data_cnn_feature_asins.npy')

    api = images_similarity(dress,bottleneck_features_train,label)
    query_random_image_no = 2000
    num_results = 5

    # api.get_similar_products_cnn(query_random_image_no,num_results)

    #function to test random image..
    query_image = './tribal_851505498.png'
    api.new_image_similar_product(query_image,5)
