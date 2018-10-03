import os
import sys
import numpy as np
from keras.utils import Sequence
import data_preprocessing
from PIL import Image

# generate batch of normalized, randomly transformed images
# data : dictionary ("filename": list of rects)


class BatchGenerator(Sequence):
    def __init__(self, data, config):
        self.data = data
        self.is_augment = config.get('is_augment', True)
        self.batch_size = config.get('batch_size', 32)
        self.grid_w = config.get('grid_w', 7)
        self.grid_h = config.get('grid_h', 7)
        self.img_w = config.get('img_w', 224)  # width of the image for NN input
        self.img_h = config.get('img_h', 224)
        self.aug_type = config.get('aug_type', 'short')
        print("iteration_num = ", self.__len__())
        if self.is_augment:
            print("use augmentation with type:", self.aug_type)

    # input : normalized rects (x_c, y_c, w, h), each value in [0,1]
    # output : labels for YOLO loss function, rect in each grid cell in relative to the cell coordinates
    def convert_GT_to_YOLO(self, rects):
        y_YOLO = np.zeros((self.grid_h, self.grid_w, 4 + 1))
        # import pdb
        # pdb.set_trace()
        for rect in rects:
            center_x = rect[0] * self.grid_w
            center_y = rect[1] * self.grid_h
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            center_x -= grid_x
            center_y -= grid_y
            center_w = rect[2] * self.grid_w
            center_h = rect[3] * self.grid_h
            y_YOLO[grid_y, grid_x, :] = np.array([1, center_x, center_y, center_w, center_h])
        return y_YOLO

    def __len__(self):
        return int(np.ceil(float(len(self.data)/self.batch_size)))

    def __getitem__(self, idx):
        indices = np.random.choice(len(self.data), self.batch_size, replace=False)
        instance_count = 0

        x_batch = np.zeros((self.batch_size, self.img_h, self.img_w, 3))                         # input images
        y_batch = np.zeros((self.batch_size, self.grid_h,  self.grid_w, 4+1))                # desired network output

        for index in indices:
            filename = list(self.data.keys())[index]
            # augment input image and fix object's position and size
            image = np.array(Image.open(filename))
            rects = self.data[filename]
            if self.is_augment:
                image, rects = data_preprocessing.random_transform(
                    image, rects, data_preprocessing.get_image_data_generator(self.aug_type))

            image_norm, rects_norm = data_preprocessing.normalize_data(image, self.img_w, self.img_h, rects)

            x_batch[instance_count] = image_norm

            y_YOLO = self.convert_GT_to_YOLO(rects_norm)

            y_batch[instance_count] = y_YOLO

            # increase instance counter in current batch
            instance_count += 1

        return x_batch, y_batch

    def on_epoch_end(self):
        print("epoch is finished")
        # np.random.shuffle(self.data.keys)
