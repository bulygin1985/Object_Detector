import numpy as np
from PIL import Image, ImageDraw

# input - image as numpy aray and rectangles on image, Normalized - each value in (0,1)
# output - image with  drawn rects as numpy array
def draw_image_rects(image, rects, is_normalized = False):
    image_to_show = image.copy()
    rects_to_show = []
    #copy rects to avoid their changing + (x_c ,y_c, w, h) -> (x, y, w, h)
    for rect in rects: 
        if is_normalized:
            rect_ = np.array( [ rect[0] - rect[2] / 2.0, rect[1] - rect[3] / 2.0, rect[2], rect[3] ] )
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
        #rect = np.array(rect, dtype=np.uint8)
        draw.rectangle(((rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3])), outline = 'red')
    return np.array(im)

#draw rects (x1, y1, x2, y2) on image
def draw_image_pred_GT(image, rects_GT, rects_pred = []):
    image_to_show = image.copy()
    image_to_show = np.array(image_to_show, dtype=np.uint8)
    im = Image.fromarray(image_to_show)
    draw = ImageDraw.Draw(im)
    for rect in rects_GT:
        draw.rectangle(((rect[0], rect[1]), (rect[2], rect[3])), outline = 'black')
    for rect in rects_pred:
        draw.rectangle(((rect[0], rect[1]), (rect[2], rect[3])), outline = 'red')
    return np.array(im)