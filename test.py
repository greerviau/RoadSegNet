import argparse
from dataset import Comma10KDataset, NUM_CLASSES
from model import SegNet
import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from constants import *

# Arguments
parser = argparse.ArgumentParser(description='Test SegNet model')

parser.add_argument('--data_root', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
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

def test():

    model.eval()

    for batch_idx, batch in enumerate(test_dataloader):
        input_tensor = torch.autograd.Variable(batch['image'])
        target_tensor = torch.autograd.Variable(batch['mask'])

        if CUDA:
            input_tensor = input_tensor.cuda(GPU_ID)
            target_tensor = target_tensor.cuda(GPU_ID)

        predicted_tensor, softmaxed_tensor = model(input_tensor)

        for idx, predicted_mask in enumerate(softmaxed_tensor):
            target_mask = target_tensor[idx]
            input_image = input_tensor[idx]

            fig = plt.figure(figsize=(16,16))

            a = fig.add_subplot(3,1,1)
            plt.imshow(input_image.detach().cpu().transpose(0, 2))
            plt.axis('off')
            a.set_title('Input Image')

            a = fig.add_subplot(3,1,2)
            predicted_mx = predicted_mask.detach().cpu().numpy()
            predicted_mx = predicted_mx.argmax(axis=0).transpose(1,0)
            predicted_mx_color = map_predictions_to_color(predicted_mx)
            plt.imshow(predicted_mx_color)
            plt.axis('off')
            a.set_title('Predicted Mask')

            a = fig.add_subplot(3,1,3)
            target_mx = target_mask.detach().cpu().transpose(1,0)
            target_mx_color = map_predictions_to_color(target_mx)
            plt.imshow(target_mx_color)
            plt.axis('off')
            a.set_title('Ground Truth')

            fig.tight_layout()

            fig.savefig(os.path.join(OUTPUT_DIR, 'prediction_{}_{}.png'.format(batch_idx, idx)))

            plt.close(fig)

if __name__ == '__main__':
    data_root = args.data_root
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    SAVED_MODEL_PATH = args.model_path
    
    OUTPUT_DIR = args.output_dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu
    
    test_dataset = Comma10KDataset(img_dir=img_dir,mask_dir=mask_dir,resolution=RESOLUTION)

    test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)

        class_weights = 1.0/test_dataset.get_class_probability().cuda(GPU_ID)
        torch.save(class_weights, 'class_weights_gpu.pth')
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,output_channels=NUM_OUTPUT_CHANNELS)

        class_weights = 1.0/test_dataset.get_class_probability()
        torch.save(class_weights, 'class_weights_cpu.pth')
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model.load_state_dict(torch.load(SAVED_MODEL_PATH))

    test()