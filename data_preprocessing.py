import numpy as np
import math
import os
import sklearn.utils
import sklearn.model_selection
from PIL import ImageFile
from PIL import Image as pImage


ImageFile.LOAD_TRUNCATED_IMAGES = True



#loads and separates images in traind and test arrays, while also optionally limiting the number of images per each class
#raises ValueError when class_size_limit and data_split are incorrectly inserted
def process_images_npy(original_data_path,    # directory where the images are originally stores
                       dir_to_save,           # directory where to store the numpy arrays
                       image_resolution,      # tuple containing the resolution in which to save the images
                       class_size_limit=None, # optional argument = if specified limits the number of images loaded per class, if this number is higher than the size of some classes, in that class only the number of available files is loaded
                       data_split = 0.9):     #  #percentage between 0 and 1 of images to store in the train array

    if class_size_limit is not None and class_size_limit < 0:
        raise ValueError("class_size_limit must be > 0")

    if data_split<0 or data_split>1:
        raise ValueError("data_split must be between 0 and 1")

    for res in image_resolution:
        if res < 0 :
            raise ValueError("both height and width must be > 0")


    class_list = os.listdir(original_data_path)
    total_number_of_files =0
    for root, dirs, files in os.walk(original_data_path):

        if class_size_limit is not None:
            total_number_of_files += min(class_size_limit,len(files))
        else:
            total_number_of_files += len(files)



    total_image_count =0
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for class_name in class_list:
        class_num = class_list.index(class_name)
        class_path = os.path.join(original_data_path,class_name)

        class_data_container = []#stores the images for this style, is later appended to the real data array

        #defining class limit for each specific class, because we may have an unbalanced dataset
        if class_size_limit is not None:
            specific_class_limit = min(class_size_limit, len(os.listdir(class_path)))
        else:
            specific_class_limit = len(os.listdir(class_path))

        train_size = math.ceil(data_split*specific_class_limit)
        test_size = specific_class_limit - train_size


        style_count = 0
        for img_name in os.listdir(class_path):
            if style_count>=specific_class_limit:
                break

            image_data = pImage.open(os.path.join(class_path, img_name))

            image_data = image_data.resize((image_resolution[0], image_resolution[1]))

            image_data = [np.asarray(image_data)]
            class_data_container= class_data_container + image_data

            total_image_count+=1
            style_count+=1
            print(total_image_count, "/", total_number_of_files)

        #Shuffling the images to mix the images inside the same class
        style_data_container = sklearn.utils.shuffle(class_data_container)
        #Splitting in train and test data
        train_data =train_data + style_data_container[:train_size]
        train_labels = train_labels + [class_num]*train_size

        test_data = test_data + style_data_container[train_size:]
        test_labels = test_labels + [class_num]*test_size

    #saving the arrays

    train_data = sklearn.utils.shuffle(train_data, random_state=0)
    train_labels = sklearn.utils.shuffle(train_labels, random_state=0)

    # random_state- seed, the same seed is used to shuffle the data and the labels so that each label still belongs to
    # the correct image
    test_data = sklearn.utils.shuffle(test_data, random_state=1)
    test_labels = sklearn.utils.shuffle(test_labels, random_state=1)
    print(len(train_data), len(test_data))
    try:
        os.mkdir(dir_to_save)
    except FileExistsError:
        pass

    np.save(os.path.join(dir_to_save, 'train_data'), train_data)
    np.save(os.path.join(dir_to_save, 'test_data'), test_data)
    np.save(os.path.join(dir_to_save, 'train_labels'), train_labels)
    np.save(os.path.join(dir_to_save, 'test_labels'), test_labels)



#loads our saved numpy arrays
def open_npy_data(data_dir, normalize = False, to_categorical = False):
    train_data = np.load(os.path.join(data_dir, "train_data.npy"))

    test_data = np.load(os.path.join(data_dir, "test_data.npy"))


    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))


    return train_data, train_labels, test_data, test_labels

#calculates the class weight for each of the classes of our unbalanced dataset
def compute_class_weight(labels):
    class_weight_dict = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                        classes=np.unique(labels),
                                                                        y=labels)
    return class_weight_dict
