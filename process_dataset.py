import numpy as np
import os
import sys
import cv2
import time
import queue
import argparse
from tqdm import tqdm
import threading
from constants import *

parser = argparse.ArgumentParser(description='Preprocess SegNet Data')

parser.add_argument('--data_root', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--pro_mask_dir', required=True)

args = parser.parse_args()

data_root = sys.argv[1]
mask_dir = os.path.join(args.data_root, args.mask_dir)
pro_mask_dir = os.path.join(args.data_root, args.pro_mask_dir)

if not os.path.exists(pro_mask_dir):
    os.makedirs(pro_mask_dir)

def process_mask(path):
    raw_mask = cv2.imread(os.path.join(mask_dir, path))
    raw_mask = cv2.cv2.cvtColor(raw_mask, cv2.COLOR_BGR2RGB)
    raw_mask = cv2.resize(raw_mask, RESOLUTION, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)

    pro_mask = np.apply_along_axis(lambda rgb: CLASS_VALUES[tuple(rgb)], 2, raw_mask)
    '''
    complete_mask = np.zeros((h,w,NUM_CLASSES))
    for i in range(NUM_CLASSES):
        complete_mask[:,:,i][pro_mask == i] = 1

    #print(complete_mask.shape)
    '''

    np.save(os.path.join(pro_mask_dir, path.replace('.png', '.npy')), pro_mask)

class Data(object):
    def __init__(self):
        self.masks = os.listdir(mask_dir)
        self.total_masks = len(self.masks)

    def get_next(self):
        return self.masks.pop(0)

    def is_empty(self):
        return len(self.masks) <= 0

    def completed(self):
        return self.total_masks - len(self.masks)

    def total(self):
        return self.total_masks

mutex = threading.Lock()
data = Data()
stop = False
number_of_threads = 12
start = time.time()

def process_mask_list(data, threadId):
    while True:
        if stop:
            break
        path = None
        with mutex:
            if data.is_empty():
                break
            path = data.get_next()
            time_elapsed = time.time() - start
            min, sec = divmod(time_elapsed, 60)
            hour, min = divmod(min, 60)
            print('thread {:02d} - mask {}/{} - time elapsed {:02d}:{:02d}:{:02d} - processing {}'.format(threadId, data.completed() , data.total(), int(hour), int(min), int(sec), path))
        process_mask(path)

# Single threaded
#process_mask_list(data)

#Multi threaded
threads = []
for i in range(number_of_threads):
    threads.append(threading.Thread(target=process_mask_list, args=(data, i+1)))
    threads[i].start()

while 1:
    try:
        time.sleep(10)
    except:
        stop = True
        sys.exit(0)