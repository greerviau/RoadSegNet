from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ROAD_CLASSES = ('road', 'lane markings', 'undrivable', 'movable', 'my car')
CLASS_VALUES = {(64, 32, 32) : 0, 
                (255, 0, 0) : 1, 
                (128, 128, 96) : 2, 
                (0, 255, 102) : 3, 
                (204, 0, 255) : 4}

NUM_CLASSES = len(ROAD_CLASSES)

RESOLUTION = (400,300)

class Comma10KDataset(Dataset):
    def __init__(self, img_dir, mask_dir, resolution, transform=None):
        self.resolution = resolution
        self.masks = os.listdir(mask_dir)
        self.transform = transform

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        name = self.masks[index]
        image_path = os.path.join(self.image_root_dir, name.replace('.npy','.png'))
        mask_path = os.path.join(self.mask_root_dir, name)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {'image': torch.FloatTensor(image),'mask' : torch.LongTensor(gt_mask)}

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in tqdm(self.masks):
            mask_path = os.path.join(self.mask_root_dir, name)

            mask = np.load(mask_path)

            h, w = mask.shape
            imx_t = np.array(mask).reshape(w*h)

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = cv2.imread(path)
        raw_image = cv2.cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = cv2.resize(raw_image, self.resolution)
        raw_image = np.transpose(raw_image, (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32)/255.0

        return imx_t

    def load_mask(self, path=None):
        mask = np.load(path)
        mask = np.transpose(mask, (1,0))
        imx_t = np.array(mask)

        return imx_t


if __name__ == '__main__':
    data_root = sys.argv[1]
    img_dir = os.path.join(data_root, 'imgs')
    mask_dir = os.path.join(data_root, 'pro_masks')


    objects_dataset = Comma10KDataset(img_dir=img_dir, mask_dir=mask_dir, resolution = RESOLUTION)

    print(objects_dataset.get_class_probability())

    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']

    image.transpose_(0, 2)
    mask.transpose_(1, 0)
    print(image.shape)
    print(mask.shape)
    
    fig = plt.figure()

    a = fig.add_subplot(1,2,1)
    plt.imshow(image)

    a = fig.add_subplot(1,2,2)
    plt.imshow(mask)

    plt.show()