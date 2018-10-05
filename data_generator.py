import numpy as np
from keras.utils import Sequence
import data_preprocessing
from PIL import Image


class BatchGenerator(Sequence):
    '''
    generate batch of normalized, randomly transformed images
    '''
    def __init__(self, data, config):
        '''
        Args:
            data : dictionary ("filename": list of rects)
            config : paarmeters for data generator as dictionary
        '''
        self.data = data
        self.filenames = list(self.data.keys())
        self.is_augment = config.get('is_augment', True)
        self.batch_size = config.get('batch_size', 32)
        self.grid_w = config.get('grid_w', 7)
        self.grid_h = config.get('grid_h', 7)
        self.img_w = config.get('img_w', 224)  # width of the image for NN input
        self.img_h = config.get('img_h', 224)
        self.shift_x = config.get('shift_x', 0.1)
        self.shift_y = config.get('shift_y', 0.1)
        self.flip = data_preprocessing.RandomHorizontalFlip()
        self.shift = data_preprocessing.RandomShift(self.shift_x, self.shift_y)
        if self.is_augment:
            print("data will be augmented with random flip and random left-right shift on " +
                  str(self.shift_x * 100.0) + " percents and top-bottom shift on " +
                  str(self.shift_y * 100.0) + " percents.")
        print("iteration_num = ", self.__len__())

    def filter_rects(self, rects_aug, width, height):
        '''
        filter removes rectangles which centers are outside the image
        and cut rectangles which go out the image
        '''
        # import pdb
        # pdb.set_trace()
        rects_aug_filtered = []
        for rect in rects_aug:
            x_c = rect[0] + rect[2] / 2.0
            y_c = rect[1] + rect[3] / 2.0
            x_max = rect[0] + rect[2]
            y_max = rect[1] + rect[3]
            w = width
            h = height
            if x_c < w and x_c > 0 and y_c < h and y_c > 0:  # at least half of image must be on image
                x_min = np.maximum(0, rect[0])
                y_min = np.maximum(0, rect[1])
                x_max = np.minimum(w-1, x_max)
                y_max = np.minimum(h-1, y_max)
                rects_aug_filtered += [np.array([x_min, y_min, x_max - x_min, y_max - y_min])]
        return rects_aug_filtered

    # input : normalized rects (x_c, y_c, w, h), each value in [0,1]
    # output : labels for YOLO loss function, rect in each grid cell in relative to the cell coordinates
    def convert_GT_to_YOLO(self, rects):
        y_YOLO = np.zeros((self.grid_h, self.grid_w, 4 + 1))
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
        return self.get_XY(indices)

    def get_XY(self, indices):
        x_batch = np.zeros((len(indices), self.img_h, self.img_w, 3))          # input images
        y_batch = np.zeros((len(indices), self.grid_h,  self.grid_w, 4+1))     # desired network output
        for i, index in enumerate(indices):
            image, rects = self.get_image_rects(index)

            image_norm, rects_norm = data_preprocessing.normalize_data(image, self.img_w, self.img_h, rects)

            x_batch[i] = image_norm

            y_YOLO = self.convert_GT_to_YOLO(rects_norm)

            y_batch[i] = y_YOLO

        return x_batch, y_batch

    def get_image_rects(self, index):
        filename = self.filenames[index]
        # augment input image and fix object's position and size (before normalization)
        image = np.array(Image.open(filename))
        rects = self.data[filename]
        if self.is_augment:
            # augment input image and fix object's rectangle
            height, width = image.shape[:2]
            image, rects = self.flip(image, rects)
            image, rects = self.shift(image, rects)
            rects = self.filter_rects(rects, width, height)
        return image, rects

    def on_epoch_end(self):
        print("epoch is finished")
        # np.random.shuffle(self.data.keys)
