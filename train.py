from __future__ import print_function
import argparse
from dataset import Comma10KDataset, NUM_CLASSES
from model import SegNet
import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from constants import *

# Arguments
parser = argparse.ArgumentParser(description='Train SegNet model')

parser.add_argument('--data_root', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--checkpoint')
parser.add_argument('--gpu', type=int)

args = parser.parse_args()

def train():
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in range(MAX_EPOCHS):
        loss_f = 0

        p_bar = tqdm(train_dataloader)
        for batch in p_bar:
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])

            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            predicted_tensor, softmaxed_tensor = model(input_tensor)

            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_tensor)
            loss.backward()
            optimizer.step()

            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()

            p_bar.set_description('Epoch: {}/{} | Loss: {:.2f}'.format(epoch+1, MAX_EPOCHS, loss_f))

        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_best.pth'))


if __name__ == '__main__':
    data_root = args.data_root
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu
    
    train_dataset = Comma10KDataset(img_dir=img_dir,mask_dir=mask_dir,resolution=RESOLUTION)

    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)

    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)
        class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,output_channels=NUM_OUTPUT_CHANNELS)
        class_weights = 1.0/train_dataset.get_class_probability()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    train()