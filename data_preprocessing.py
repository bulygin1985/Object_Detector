# create image data generator objects
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image, ImageDraw
import pdb

def get_image_data_generator():
    image_data_generator = ImageDataGenerator(
        horizontal_flip=True,   # randomly flip images
        rotation_range = 10,    # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
        shear_range=0.1,
        zoom_range=0.1
    )
    # image_data_generator = ImageDataGenerator(horizontal_flip=True, shear_range=0.1, zoom_range=0.1)
    return image_data_generator

# image dtype=np.uint8 !!!
# input: objects bounding boxes : (x_min, y_min, width, height)
# output : augmented image, and list of augmented bounding boxes (x_min, y_min, width, height)
def random_transform(image, rects, image_data_generator):
    image = np.array(image, dtype=np.uint8)
    seed = np.random.randint(10000)
    image_aug = image_data_generator.random_transform(image, seed=seed).copy()
    fill_mode = image_data_generator.fill_mode
    
    image_data_generator.fill_mode = 'constant'
    rects_aug = []
    for rect in rects:
        b = np.array([int(rect[0]), int(rect[1]), int(rect[0] + rect[2]), int(rect[1] + rect[3]) ] )
        assert(b[0] < b[2] and b[1] < b[3]), 'Annotations contain invalid box: {}'.format(b)
        assert(b[2] < image.shape[1] and b[3] < image.shape[0]), 'Annotation ({}) is outside of image shape ({}).'.format(b, image.shape)
        mask = np.zeros_like(image_aug, dtype=np.uint8)
        mask[b[1]:b[3], b[0]:b[2], :] = 255
        mask = image_data_generator.random_transform(mask, seed=seed)[..., 0]
        
        [i, j] = np.where(mask == 255)
        b_aug = np.array([float(min(j)), float(min(i)), float(max(j)), float(max(i))])
        rects_aug += [np.array([ b_aug[0], b_aug[1], b_aug[2] - b_aug[0], b_aug[3] - b_aug[1] ])]
    # restore fill_mode
    image_data_generator.fill_mode = fill_mode
    return image_aug, rects_aug

# image (after augmentation) is raw image as numpy array
# (input_w, input_h) - image input size for NN
# rect - raw rect (after augmentation)
# 1) image value from (0, 255) -> (0,1)
# 2) rescale image to the specified size (img_w, img_h)
# 3) change bounding box in according to image rescaling
# 4) bounding box (x,y,w,h) -> (x_c, y_c, w, h), each dim in (0,1)
def normalize_data(image, input_w, input_h, rects):
    im_w = image.shape[1]  #num of columns
    im_h = image.shape[0]  #num of rows
    image = np.array(image, dtype=np.uint8)
    image_PIL = Image.fromarray(image).resize((input_w, input_h), Image.ANTIALIAS) #resize image using PIL
    image = np.array(image_PIL, dtype=np.float)

    # image = np.array(Image.fromarray(image).resize((input_w, input_h), Image.ANTIALIAS))

    #rectangle center in relative coor in [0,1]
    rects_norm = []
    for rect in rects:
        center = np.array( [ (float(rect[0]) + float(rect[2]) / 2.0) / im_w, (float(rect[1]) + float(rect[3]) / 2.0) / im_h ] )
        rect = np.array([center[0], center[1], float(rect[2]) / im_w, float(rect[3]) / im_h])
        rects_norm += [rect]
    image /= 255.0
    return image, rects_norm