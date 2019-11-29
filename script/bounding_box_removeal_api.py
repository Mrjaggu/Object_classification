from utils import *
import imutils

class bounding_box():

    def __init__(self,source,save_path):
        self.source = source
        self.save_path = save_path

    def crop_save(self):
        print('[Info] Loading images...')
        folder = os.listdir(self.source)
        for image_folder in tqdm(folder):
            for id,images in tqdm(enumerate(os.listdir(self.source + '/' + image_folder))):
                image_name = images.split('.')[0]
                image_path = self.source + '/' + image_folder + '/' + images
                image = cv2.imread(image_path)
                hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV_FULL)
                vw=255*np.uint8((hsv[:,:,0]==0)&(hsv[:,:,1]==255)&(hsv[:,:,2]==255))
                cnts = cv2.findContours(vw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                for contours in cnts:
                    x,y,w,h = cv2.boundingRect(np.array(contours))
                    im2 = image[y:y+h,x:x+w]

                    cropped_image_path = self.save_path + '/' +  image_folder
                    if not os.path.exists(cropped_image_path):
                      os.makedirs(cropped_image_path)

                    cv2.imwrite(cropped_image_path + '/' + str(image_name) + '.png',im2)
        print('[Info] All cropped image are saved....')

if __name__ == '__main__':
    source = '/home/ajay/GreenDeck/zip'
    destination_path = '/home/ajay/GreenDeck/new_image'
    api = bounding_box(source,destination_path)
    api.crop_save()
