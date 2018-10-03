import numpy as np
import tensorflow as tf


class Encoder:
    def __init__(self, GRID_W, GRID_H, input_w, input_h, threshold=0.5):
        self.GRID_W = GRID_W
        self.GRID_H = GRID_H
        self.input_w = input_w
        self.input_h = input_h
        self.threshold = threshold
    # input  - NN output for single image (relative to cell coordinates)
    # output - rects with score > T in relative to image coordinates

    def label_to_rects(self, out):
        scores = out[:, :, 0].copy()
        # pdb.set_trace()
        rects = out[:, :, 1:].copy()
        indices = np.where(scores > self.threshold)
        rects[:, :, 0] += np.array(range(self.GRID_W)).reshape(1, self.GRID_W)
        rects[:, :, 1] += np.array(range(self.GRID_H)).reshape(self.GRID_H, 1)
        scores_filtered = scores[indices]
        rects_filtered = rects[indices]
        rects_filtered[:, 0] /= float(self.GRID_W)
        rects_filtered[:, 1] /= float(self.GRID_H)
        rects_filtered[:, 2] /= float(self.GRID_H)
        rects_filtered[:, 3] /= float(self.GRID_W)
        return rects_filtered, scores_filtered

    def relative_to_absolute(self, rects):
        rects[:, 0] *= self.input_w
        rects[:, 1] *= self.input_h
        rects[:, 2] *= self.input_w
        rects[:, 3] *= self.input_h
        return rects

    # convert (x_c, y_c, w, h) to (x1, y1, x2. y2)
    def convert(self, rects):
        rects[:, 0] -= rects[:, 2] / 2.0
        rects[:, 1] -= rects[:, 3] / 2.0
        rects[:, 2] += rects[:, 0]
        rects[:, 3] += rects[:, 1]
        return rects

    def encode(self, nn_out):
        rects = []
        scores = []
        for out in nn_out:
            rects_pred, scores_pred = self.label_to_rects(out)
            self.relative_to_absolute(rects_pred)
            self.convert(rects_pred)
            indices = []
            with tf.Session() as non_max_supression_session:
                nms_indices = tf.image.non_max_suppression(
                    rects_pred, scores_pred, max_output_size=10, iou_threshold=0.5)
                indices = nms_indices.eval()
            rects += [rects_pred[indices]]
            scores += [scores_pred[indices]]
        return rects, scores

    def encode_GT(self, nn_out):
        rects = []
        for out in nn_out:
            rects_GT, _ = self.label_to_rects(out)
            self.relative_to_absolute(rects_GT)
            self.convert(rects_GT)
            rects += [rects_GT]
        return rects
