import logging
import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
from tqdm import tqdm

from smdebug.pytorch import get_hook
from smdebug.pytorch import modes

# Set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

hook = get_hook(create_if_not_exists=True)
class CustomDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, images_dict, labels_dict=None, transform=None):
        self.images_dict = images_dict
        self.labels_dict = labels_dict
        self.image_ids = list(images_dict.keys)
        self.transform = transform

    def __len__(self):
        
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = self.images_dict[image_id]

        # Transform np array to PIL image
        image = Image.fromarray(image)

        # Apply any requires transformations if any
        if self.transform:
            image = self.transform(image)

        if self.labels_dict is not None:
            label = self.labels_dict.get(image_id, None)
            if label is None:
                raise ValueError(f"Label not found for image ID: {image_id}")
            return image, label
        else:
            return image
        
def test(model, test_loader, loss_criterion):
    '''Define testing loop, return the test accuray/loss of the model.
    Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Testing started.")
    # Set model to test mode
    model.eval()
    test_loss = 0
    correct = 0

    # With model set to test mode, iterate through data
    with torch.no_grad():
        #1 Loop through data
        for (data, target) in test_loader:
            # 2 Forward pass
            preds = model(data)
            #3. Compute loss
            loss = loss_criterion(preds, target)
            test_loss += loss.item()

            # Count the number of correct predictions
            # Get the index of the maximum probability class
            pred = preds.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate avg test loss
    test_loss /= len(test_loader.dataset)
    # Log testing metrics
    logger.info(f"Test loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset)}%)\n")

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    pass

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    
    args=parser.parse_args()
    
    main(args)
