import numpy as np
from PIL import Image, ImageDraw

# input - image as numpy aray and rectangles on image, Normalized - each value in (0,1)
# output - image with  drawn rects as numpy array


def draw_image_rects(image, rects, is_normalized=False):
    image_to_show = image.copy()
    rects_to_show = []
    # copy rects to avoid their changing + (x_c ,y_c, w, h) -> (x, y, w, h)
    for rect in rects:
        if is_normalized:
            rect_ = np.array([rect[0] - rect[2] / 2.0, rect[1] - rect[3] / 2.0, rect[2], rect[3]])
            rects_to_show += [rect_]
        else:
            rects_to_show += [rect.copy()]
    if is_normalized:
        image_to_show *= 255.0
        for rect in rects_to_show:
            rect[0] *= float(image.shape[1])
            rect[1] *= float(image.shape[0])
            rect[2] *= float(image.shape[1])
            rect[3] *= float(image.shape[0])
            rect = np.array(rect, dtype=np.uint8)
    image_to_show = np.array(image_to_show, dtype=np.uint8)
    im = Image.fromarray(image_to_show)
    draw = ImageDraw.Draw(im)
    for rect in rects_to_show:
        draw_rectangle(draw, rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3], color='red', width=3)
    return np.array(im)


def draw_rectangle(draw, x_min, y_min, x_max, y_max, color='black', width=3):
    '''
    draw rects with width > 1, because PIL cannot do it
    '''
    shift = int(width/2)
    for i in range(-shift, shift):
        draw.rectangle(((x_min - i, y_min - i), (x_max + i, y_max + i)), outline=color)


def draw_image_pred_GT(image, rects_GT=[], rects_pred=[]):
    '''
    draw rects (x1, y1, x2, y2) on image
    '''
    image_to_show = image.copy()
    image_to_show = np.array(image_to_show, dtype=np.uint8)
    im = Image.fromarray(image_to_show)
    draw = ImageDraw.Draw(im)
    for rect in rects_GT:
        draw_rectangle(draw, rect[0], rect[1], rect[2], rect[3], color='green', width=3)
        # draw.rectangle(((rect[0], rect[1]), (rect[2], rect[3])), outline='green')
    for rect in rects_pred:
        draw_rectangle(draw, rect[0], rect[1], rect[2], rect[3], color='red', width=3)
        # draw.rectangle(((rect[0], rect[1]), (rect[2], rect[3])), outline='red')
    return np.array(im)
