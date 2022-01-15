import numpy as np
import pandas as pd

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize

#paths
input_root = './'
test_files = f"{input_root}/test/test_60.71/"
train_files = f"{input_root}/train/train_60.71/"
val_files = f"{input_root}/validation/validation_60.71/"


#getting names of classes 
def get_class_labels():
    class_labels = pd.DataFrame(columns=['class'])
    for category in os.listdir(test_files):
        n_test = len(os.listdir(f"{test_files}{category}"))
        n_train = len(os.listdir(f"{train_files}{category}"))
        n_val = len(os.listdir(f"{val_files}{category}"))
        class_labels = class_labels.append({'class': category, 'n_test': n_test, 'n_train': n_train, 'n_val': n_val}, ignore_index=True)
    class_labels['class_id'] = class_labels.index
    return class_labels


def get_images(source_dir, class_labels=None):
    if class_labels == None:
        class_labels = get_class_labels()

    data = pd.DataFrame(columns=['class', 'path'])
    for dirname in os.listdir(source_dir):
        for filename in os.listdir(f"{source_dir}{dirname}"):
            data = data.append({'class': dirname, 'path': f"{source_dir}/{dirname}/{filename}"}, ignore_index=True)
    
    data = data.merge(class_labels, on='class', how='left')
    
    y = np.array(data['class_id'])
    x = [iio.imread(path)[:,:,:3] for path in tqdm(data['path'].to_list())]

    return x, y, data


def resize_images(source_dir, dest_dir, new_size):
    data = pd.DataFrame(columns=['class', 'path'])
    for dirname in tqdm(os.listdir(source_dir)):
        os.mkdir(f"{dest_dir}{dirname}")
        for filename in tqdm(os.listdir(f"{source_dir}{dirname}")):
            path_in = f"{source_dir}{dirname}{filename}"
            path_out = f"{dest_dir}{dirname}{filename}"
            x = resize(iio.imread(path_in)[:,:,:3], new_size)
            iio.imwrite(x, path_out)