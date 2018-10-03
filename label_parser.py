import numpy as np
import csv
import json

# csv annotation


def parse_annotation(path, N=0):
    data = {}  # each element contains image filename and list of rects
    with open(path + "label.csv") as label_file:
        reader = csv.reader(label_file)
        for idx, row in enumerate(reader):
            filename = path + "/" + row[0]
            if filename not in data:
                data[filename] = []
            data[filename] += [np.array([float(row[1]), float(row[2]), float(row[3]), float(row[4])])]
            if N > 0 and idx >= N:
                break
    return data


def parse_annotation_json(path, annotation_file_name='label.json', N=0):
    with open(path + annotation_file_name) as json_file:
        data_short = {}
        data = json.load(json_file)
        for idx, filename in enumerate(data.keys()):
            full_path = path + "/" + filename
            data_short[full_path] = []
            for rect in data[filename]:
                data_short[full_path] += [np.array(rect)]
            if N > 0 and idx >= N:
                break
        return data_short
