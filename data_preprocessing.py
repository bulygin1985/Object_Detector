import numpy as np
from PIL import Image


class HorizontalFlip:
    '''
    Create horizontally flipped image and label
    '''

    def __call__(self, image, labels=None):
        '''
        Args:
            image : numpy array of shape (width, height, channel_num)
            label (list of floats): location of predicted object as
                [xmin, ymin, width, height]

        Returns:
            float: value of the IoU for the two boxes.
        '''
        img_height, img_width = image.shape[:2]
        image = image[:, ::-1]
        if labels is None:
            return image
        else:
            labels = np.copy(labels)
            labels[:, [0]] = img_width - labels[:, [0]] - labels[:, [2]]
            return image, labels


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.flip = HorizontalFlip()

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            return self.flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels


class Shift:
    '''
    Shift image and lables horizontally and vertically
    '''

    def __init__(self, shift_x=0.1, shift_y=0.1):
        # print("shift_x = ", shift_x, "shift_y = ", shift_y)
        self.shift_x = shift_x
        self.shift_y = shift_y

    def transpose_labels(self, labels):
        labels_transposed = np.zeros_like(labels)
        labels_transposed[:, 0] = labels[:, 1]
        labels_transposed[:, 1] = labels[:, 0]
        labels_transposed[:, 2] = labels[:, 3]
        labels_transposed[:, 3] = labels[:, 2]
        return labels_transposed

    def horizontal_shift(self, image, shift_percent, labels=None):
        img_height, img_width, channel_num = image.shape
        shift = int(shift_percent * img_width)
        shifted_image = np.zeros_like(image)
        if shift > 0:
            shifted_image[:, shift:] = image[:, :-shift]
            first_col = image[:, 0]  # 'black' part of image is replaced by the first row
            shifted_image[:, :shift] = np.tile(first_col, shift).reshape(-1, shift, channel_num)
        elif shift < 0:
            shift_ = -shift
            shifted_image[:, :-shift_] = image[:, shift_:]
            last_col = image[:, -1]  # 'black' part of image is replaced by the last row
            shifted_image[:, -shift_:] = np.tile(last_col, shift_).reshape(-1, shift_, channel_num)
        elif shift == 0:
            shifted_image = image

        if labels is None:
            return shifted_image
        else:
            labels = np.copy(labels)
            labels[:, [0]] = labels[:, [0]] + shift
            return shifted_image, labels

    def __call__(self, image, labels=None):
        if labels is None:
            image_shifted_x = self.horizontal_shift(image, self.shift_x)
            image_shifted_transposed = self.horizontal_shift(image_shifted_x.transpose(1, 0, 2), self.shift_y)
            return image_shifted_transposed.transpose(1, 0, 2)
        else:
            image_shifted_x, labels = self.horizontal_shift(image, self.shift_x, labels)
            image_shifted_transposed, labels_shifted_transposed = self.horizontal_shift(
                image_shifted_x.transpose(1, 0, 2), self.shift_y, self.transpose_labels(labels))
            return image_shifted_transposed.transpose(1, 0, 2), self.transpose_labels(labels_shifted_transposed)


class RandomShift:
    '''
    Shift image and lables horizontally and vertically
    on random part in interval x in (-shift_x_max, shift_x_max),
    y in (-shift_y_max, shift_y_max)
    '''

    def __init__(self, shift_x_max=0.1, shift_y_max=0.1):
        if shift_x_max < 0 or shift_y_max < 0 or shift_x_max > 1 or shift_y_max > 1:
            raise ValueError("Requirements : shift_x_max and shift_y_max are in (0,1)")
        self.shift_x_max = shift_x_max
        self.shift_y_max = shift_y_max

    def __call__(self, image, labels=None):
        mult_x, mult_y = np.random.rand(2) * 2 - 1  # random number in the interval (-1,1)
        shift = Shift(mult_x * self.shift_x_max, mult_y * self.shift_y_max)
        if labels is None:
            return shift(image)
        else:
            return shift(image, labels)

# image (after augmentation) is raw image as numpy array
# (input_w, input_h) - image input size for NN
# rect - raw rect (after augmentation)
# 1) image value from (0, 255) -> (0,1)
# 2) rescale image to the specified size (img_w, img_h)
# 3) change bounding box in according to image rescaling
# 4) bounding box (x,y,w,h) -> (x_c, y_c, w, h), each dim in (0,1)


def normalize_data(image, input_w, input_h, rects):
    im_w = image.shape[1]  # num of columns
    im_h = image.shape[0]  # num of rows
    image = np.array(image, dtype=np.uint8)
    image_PIL = Image.fromarray(image).resize((input_w, input_h), Image.ANTIALIAS)  # resize image using PIL
    image = np.array(image_PIL, dtype=np.float)

    # image = np.array(Image.fromarray(image).resize((input_w, input_h), Image.ANTIALIAS))

    # rectangle center in relative coor in [0,1]
    rects_norm = []
    for rect in rects:
        center = np.array([(float(rect[0]) + float(rect[2]) / 2.0) / im_w,
                           (float(rect[1]) + float(rect[3]) / 2.0) / im_h])
        rect = np.array([center[0], center[1], float(rect[2]) / im_w, float(rect[3]) / im_h])
        rects_norm += [rect]
    image /= 255.0
    return image, rects_norm
