from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_files', default='data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocotalk_box_36', help='output feature files')

args = parser.parse_args()
csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(args.downloaded_files) as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
    for item in reader:
        item['image_id'] = int(item['image_id'])
        item['num_boxes'] = int(item['num_boxes'])
        for field in ['boxes', 'features']:
            item[field] = np.frombuffer(base64.decodestring(item[field].encode()), 
                    dtype=np.float32).reshape((item['num_boxes'],-1))
        np.savez_compressed(os.path.join(args.output_dir, str(item['image_id'])), 
            feat=item['features'],
            hw=[int(item['image_h']), int(item['image_w'])],
            box=item['boxes'])
        print(str(item['image_id']))
