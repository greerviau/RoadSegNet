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

def map_predictions_to_color(pred_image):
    h, w = pred_image.shape
    color = np.zeros((h,w,3), dtype=np.uint8)
    for i in range(len(CLASS_VALUES)):
        color[pred_image == i] = list(CLASS_VALUES.keys())[i]
    return color

def inference():

    model.eval()

    img_list = os.listdir(img_dir)
    shuffle(img_list)
    i = 0
    for img_path in img_list:
        image = cv2.imread(os.path.join(img_dir, img_path))
        image = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, RESOLUTION)
        processed_img = np.transpose(image, (2,1,0))
        imx_t = np.array(processed_img, dtype=np.float32)/255.0
        image_tensor = torch.FloatTensor([imx_t])

        input_tensor = torch.autograd.Variable(image_tensor)

        if CUDA:
            input_tensor = input_tensor.cuda(GPU_ID)

        predicted_tensor, softmaxed_tensor = model(input_tensor)

        predicted_mask = softmaxed_tensor[0]

        fig = plt.figure(figsize=(16,16))

        a = fig.add_subplot(2,1,1)
        plt.imshow(image)
        plt.axis('off')
        a.set_title('Input Image')

        a = fig.add_subplot(2,1,2)
        predicted_mx = predicted_mask.detach().cpu().numpy()
        predicted_mx = predicted_mx.argmax(axis=0).transpose(1,0)
        predicted_mx_color = map_predictions_to_color(predicted_mx)
        plt.imshow(predicted_mx_color)
        plt.axis('off')
        a.set_title('Predicted Mask')

        fig.tight_layout()

        fig.savefig(os.path.join(OUTPUT_DIR, 'prediction_{}'.format(i)))
        i += 1

        plt.close(fig)

if __name__ == '__main__':
    data_root = args.data_root
    img_dir = os.path.join(data_root, args.img_dir)

    SAVED_MODEL_PATH = args.model_path
    
    OUTPUT_DIR = args.output_dir
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