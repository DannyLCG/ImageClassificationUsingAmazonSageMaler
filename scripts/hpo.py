'''This scrip serves as entrypoint to perform Hyperparameter Optimization using SageMaker's Hyperprameter Tuner.
Here we create an instance of EfficientNet_b4 and define the trainig and testing loop as well as data loaders from S3.'''
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, loss_criterion):
    '''Define testing loop, return the test accuray/loss of the model.
    Remember to include any debugging/profiling hooks that you might need
    '''
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
    logger.info(f"Test Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset)}%)\n")

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    '''Define training loop, return training and evaluation metrics.
    Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Training started.")

    # 0. Loop through each epoch
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        # Set model to training mode
        model.train()
        train_loss = 0
        # 1. Loop through dataset
        for data, target in train_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            # 2. Zero gradients
            optimizer.zero_grad()
            # 3. Forward pass
            preds = model(data)
            # 4. Compute loss
            loss = criterion(preds, target)
            # 5. Backward pass
            loss.backward()
            # 6. Update weights
            optimizer.step()

            # Update training loss/epoch
            train_loss += loss.item()

        # Log training metrics
        train_loss /= len(train_loader.dataset)
        logger.info(f"Epoch: {epoch}/{epochs}, Training Loss: {train_loss:.4f}")

        # Perform validation
        # Set model to evaluation mode
        model.eval()
        val_loss = 0

        with torch.no_grad():
            #1. Loop through data
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(device), target.to(device)
                #2 Forward pass
                preds = model(data)
                #3 Compute loss
                loss = criterion(preds, target)
                # Update validation loss
                val_loss += loss.item()

            # Log validation metrics
            val_loss /= len(val_loader.dataset)
            logger.info(f"Epoch {epoch}/{epochs}, Validation Loss: {val_loss:.4f}")

    logger.info(f"Finished training for {epochs}s...")


def net(num_classes, device):
    '''Instance the EfficientNet_b4 model'''
    model = models.efficientnet_b4(pretrained=True)
    # Freeze the network
    for param in model.parameters():
        param.requires_grad = False

    # Add a new FC layer. Newly constructed modules have 'requires_grad=True' by default
    num_feats = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_feats, num_classes)

    # Move model to available device
    model.to(device)

    return model

def read_images(file_path):
    '''Function to read image data from a HDF5 file.
    Params:
        file_path: str, the file path to the HDF5 file.
        
    return: dict, a dictionary in the form of {Image ID: numpy array}"'''
    with h5py.File(file_path, 'r') as file:
        ids_list = list(file.keys())
        images = {}
        for img_id in tqdm(ids_list):
            # Extract the image data
            image_data = file[img_id][()] #retrieve the entire data from each dataset
            image = Image.open(io.BytesIO(image_data))
            images[img_id] = np.array(image)

    return images

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''


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
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    args=parser.parse_args()
    
    main(args)
