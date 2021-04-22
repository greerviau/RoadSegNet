import argparse
from dataset import Comma10KDataset, NUM_CLASSES
from model import SegNet
import os
import cv2
import time
from random import shuffle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from constants import *

# Arguments
parser = argparse.ArgumentParser(description='SegNet model inference')

parser.add_argument('--data_root', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--gpu', type=int)

args = parser.parse_args()

def inference():

    model.eval()

    for img_path in tqdm(os.listdir(img_dir)):
        image = cv2.imread(os.path.join(img_dir, img_path))
        img_size = tuple(reversed(image.shape[:2]))

        image_pro = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pro = cv2.resize(image_pro, RESOLUTION)
        image_pro = np.transpose(image_pro, (2,1,0))
        imx_t = np.array(image_pro, dtype=np.float32)/255.0
        image_tensor = torch.FloatTensor([imx_t])

        input_tensor = torch.autograd.Variable(image_tensor)

        if CUDA:
            input_tensor = input_tensor.cuda(GPU_ID)

        predicted_tensor, softmaxed_tensor = model(input_tensor)

        predicted_mask = softmaxed_tensor[0]
        predicted_mx = predicted_mask.detach().cpu().numpy()
        predicted_mx = predicted_mx.argmax(axis=0).transpose(1,0)
        predicted_mx = cv2.resize(predicted_mx, img_size, interpolation = cv2.INTER_NEAREST)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[(predicted_mx == 3) | (predicted_mx == 4)] = 0
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_path), image)

if __name__ == '__main__':
    data_root = args.data_root
    img_dir = os.path.join(data_root, args.img_dir)

    SAVED_MODEL_PATH = args.model_path
    
    OUTPUT_DIR = os.path.join(data_root, args.output_dir)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)

        class_weights = torch.load('class_weights_gpu.pth')
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,output_channels=NUM_OUTPUT_CHANNELS)

        class_weights = torch.load('class_weights_cpu.pth')
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model.load_state_dict(torch.load(SAVED_MODEL_PATH))

    inference()