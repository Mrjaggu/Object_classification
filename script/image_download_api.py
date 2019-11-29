from utils import *

class download():
    def __init__(self,data,directory):
        self.data=data
        self.directory = directory

    def image_download(self):
        print('[Info] Downloading image...')
        for index,image_url in tqdm(enumerate(self.data['image_url'])):
            # url = image_url
            r = requests.get(image_url, allow_redirects=True)
            image_name = self.data['category'][index]
            folder_path = self.directory + '/' + image_name
            if not os.path.exists(folder_path):
              os.makedirs(folder_path)
            save_image = folder_path + '/' + image_name + '_'+str(self.data['_unit_id'][index])+ '.png'
            open(save_image, 'wb').write(r.content)
        print("[Info] Images got downloaded...")

if __name__ =='__main__':
    data_path = '/home/ajay/GreenDeck/Data/dress_patterns.csv'
    loc = '/home/ajay/GreenDeck/Image'
    data = pd.read_csv(data_path)
    api = download(data,loc)
    api.image_download()
