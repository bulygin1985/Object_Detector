import os
import numpy as np
import xml.etree.ElementTree as ET

id2name={1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'} 
name2id={'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}


class PascalDataset():
    def __init__(self, path):
        self.path = path
        ann_files_test = os.listdir(path + 'Annotations')
        self.data = {}        # image_path to list of bboxes
        for file_name in ann_files_test:
            parts = file_name.split('.')
            if 2 == len(parts) and 'xml' == parts[1]:
                filename, bboxes = self.read_content(path + 'Annotations/' + file_name)
                self.data[path + 'JPEGImages/' + filename] = bboxes
                
    def get_filename2bboxes():
        return self.data
        
    def read_content(self, xml_file: str):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        bboxes = []
        filename = root.find('filename').text
        
        for boxes in root.iter('object'):
            xmin = int(boxes.find("bndbox").find('xmin').text)
            ymin = int(boxes.find("bndbox").find('ymin').text)      
            xmax = int(boxes.find("bndbox").find('xmax').text)
            ymax = int(boxes.find("bndbox").find('ymax').text)
            name = boxes.find("name").text
            bboxes.append([name2id[name], xmin, ymin, xmax - xmin, ymax - ymin])
        return filename, bboxes
                