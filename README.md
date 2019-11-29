## Object_classification and Image similarity..

# Task 1

## "To build an image recognition model which is capable of identifying the pattern on a dress image.”

  You can get the script is in final_model folder.

## Dataset Information​ :
1. This dataset contains links to images of women's dresses. <br>
2. The corresponding images are categorized into 17 different pattern
types.<br>
3. Most pattern categories have hundreds to thousands of examples.
<br>

## Trained Model download link

1. Vgg16 -- [ [link] ](https://drive.google.com/open?id=1t1twM-v4jpiacL5UTh-th0HEd55G7-aK) <br>
2. Vgg19 -- [ [link] ](https://drive.google.com/open?id=1wyK4BfBNhy7SDVfMWRxB8hlmdJTomLd5) <br>
3. ResNet -- [ [link] ] (https://drive.google.com/open?id=1gcNiwLTGLuTI0mmeyCCw74gBUU0teZGi) <br>
4. Final best Model -- [ [link] ] (https://drive.google.com/open?id=1wyK4BfBNhy7SDVfMWRxB8hlmdJTomLd5)

### Training data

  Run the script 'train_vgg19.py'

### Testing data

  Run the script 'test_vgg19.py'


# Task 2

## Download data
1. Image fearure [https://drive.google.com/open?id=1XdPlI4o0KojNJq68T8KVMnoeJ3x8eBP4]
2. Image label [https://drive.google.com/open?id=1imZlhYykWkHnMzxL8Ec7qIvrDtziDDkG]
3. Dress data - Folder Data

## "Given a random query image we need to return the similar products.”

  You can get the script in images_similarity folder.

### To create cnn feature for data
 1. Run the script 'create_cnn_feature.py'
 2. Save the image cnn feature and it's corresponding label 

### Testing
 1. Run the script 'images_similarity.py' by passing random image and cnn image feature and label path
