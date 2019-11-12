import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import ImageFile
from PIL import Image as pImage
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
PREDICT_DIR = "paintings_to_predict"
IMG_SIZE = 180
CLASSES = ['Minimalism', 'Romanticism', 'Rococo', 'Post_Impressionism', 'Art_Nouveau_Modern', 'Renaissance', 'Pointillism', 'Realism', 'Ukiyo_e', 'Symbolism', 'Baroque', 'Cubism', 'Abstract', 'Pop_Art', 'Impressionism', 'Expressionism', 'Color_Field_Painting']


data = []
image_names = []
for image_name in os.listdir(PREDICT_DIR):
    try:
        image_data = pImage.open(os.path.join(PREDICT_DIR, image_name))
        image_data = image_data.resize((IMG_SIZE, IMG_SIZE))

        image_array = np.asarray(image_data)
        data.append(image_array)
        image_names.append(image_name)
    except IOError:
        print("Could not load '"+image_name+"', file type not supported.")

data =np.array(data, dtype=np.float32)
data = data/255 #to match how he NN was trained, with pixel values between 0 and 1
print(image_names)
if len(data)!= 0:
    try:
        model = keras.models.load_model("fine_tuned_VGG16_180x180.h5")
        predictions = model.predict(data)
        for i,prediction in enumerate(predictions): # returns both the element and an index counter
            with np.printoptions(precision=5, suppress=True):
                index_list = np.flip(np.argsort(prediction,axis= 0))
                #returns an array with the original indexes of the sorted array, that array is then inverted to
                # get the indexes of the biggest elements first
                print("\n"+image_names[i])
                print(" Most likely: ",CLASSES[index_list[0]])
                print(" Second prediction: ", CLASSES[index_list[1]])
                print(" Third prediction: ", CLASSES[index_list[2]])

    except OSError:
        print("There is no neural network in the current directory, please certify that the file 'fine_tuned_VGG16_180x180.h5' is in this directory")
else:
    print("There are no images to predict, please put them in the 'images to predict' directory")


