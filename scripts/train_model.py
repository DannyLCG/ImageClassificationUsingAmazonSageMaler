import logging
import io
import os
import sys
import time

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
from smdebug.profiler.utils import str2bool
import smdebug.pytorch as smd

# Set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Create debugger/profiler hook
hook = smd.get_hook(create_if_not_exists=True)

class CustomDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, images_dict, labels_dict=None, transform=None):
        self.images_dict = images_dict
        self.labels_dict = labels_dict
        self.image_ids = list(images_dict.keys())
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
        
def test(model, test_loader, device):
    '''Define testing loop, return the test accuray/loss of the model.
    '''
    logger.info("Testing started.")
    # ======================================================#
        # Set hook to eval mode
    # ======================================================#
    if hook:
        hook.set_mode(modes.EVAL)

    # Set model to test mode
    model.eval()


    # With model set to test mode, iterate through data
    with torch.no_grad():
        
        correct_test, total_test = 0,0 #to calculate accuracy
        epoch_times = [] # To track the time of testing
        start = time.time()

        # loop through data
        for data, label in test_loader:
            # Move data to device
            data, label = data.to(device), label.to(device)
            # Make predictions for the test data
            preds = model(data)
            _, prediction = torch.max(preds, 1) # Extract indices/labels

            # Calculate accuracy
            correct_test += (prediction == label).sum().item()
            total_test += label.size(0)

        # Log metric
        test_accuracy = correct_test / total_test
        logger.info(f"Test Accuracy: {test_accuracy:.2f}")

        # Convert predictions to to numpy
        #prediction = prediction.cpu().numpy() # Leave as array if further operations are needed  
    
    # Log predictions
    logger.info(f"Predicted classes for the test data: {prediction.tolist()}")

    # Calculate testing time (remember it depends on the batch size and size of testing data)
    epoch_time = time.time() - start
    epoch_times.append(epoch_time)
    p50 = np.percentile(epoch_times, 50)
    logger.info("Testing completed successfully.")

    return p50

def train(model, train_loader, criterion, optimizer, epochs, device):
    '''Define training loop, return training metrics
    '''
    logger.info("Training started.")
     # ====================================#
    # 1. Create the hook (created already)
    # ====================================#
    epoch_times = []

    # ======================================================#
    # 2. Set hook to track the loss 
    # ======================================================#
    if hook:
        hook.register_loss(criterion)

    # 0. Loop through each epoch
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        # ======================================================#
        # 3. Set hook to training mode
        # ======================================================#
        if hook:
            hook.set_mode(modes.TRAIN)
        start = time.time()

        # Set model to training mode
        model.train()
        train_loss = 0
        correct_train, total_train = 0, 0
    
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

            # Calculate number of correct predictions
            _, prediction = torch.max(preds, 1) # Get the index of the max log, which corresponds to the label
            correct_train += (prediction == target).sum().item()
            total_train += target.size(0)

        # Log training metrics
        train_loss /= len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        logger.info(f"Epoch: {epoch}/{epochs}")
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")

        #Calculate p50 epoch training time
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

    logger.info(f"Finished training for {epochs}s...")
    #Calculate training time
    p50 = np.percentile(epoch_times, 50)

    return p50
    
def net(device):
    '''Instance the EfficientNet_b4 model'''
    model = models.resnet50(pretrained=True)
    # Freeze the network
    for param in model.parameters():
        param.requires_grad = False

    # Add a new FC layer. Newly constructed modules have 'requires_grad=True' by default
    num_feats = model.fc.in_features
    model.fc = torch.nn.Linear(num_feats, 2)

    # Move model to available device
    model.to(device)

    return model

def find_files(data_dir, hdf5_suffix=".hdf5", csv_suffix=".csv"):
    '''
    As the channel parameter is a S3 directory, we need to retrieve the file paths for each file.
    Our data is in separate dirs for each split, but each dir has 2 files.
    Remeber that the test data does not need labels, so the CSV file is optional.
    '''
    logger.info("Finding file paths from S3.")

    hdf5_file = None
    csv_file = None

    # List files and identify hdf5 and csv files
    for file_name in os.listdir(data_dir):
        if file_name.endswith(hdf5_suffix):
            hdf5_file = os.path.join(data_dir, file_name)
        elif file_name.endswith(csv_suffix):
            csv_file = os.path.join(data_dir, file_name)

    # Handle missing HDF5 file (CSV file is optional for the test data)
    if not hdf5_file:
        raise FileNotFoundError("Required HDF5 file not found in the directory.")
    
    if not csv_file:
        logger.warning("CSV file not found. Proceeding without it.")

    return hdf5_file, csv_file

def load_data(hdf5_path, csv_path=None):
    '''Function to load image data from a HDF5 and load labels from a CSV file '''
    logger.info("Started to load data from files...")

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        keys = list(hdf5_file.keys())
        images_dict = {}

        for key in keys:
            image_data = hdf5_file[key][()]
            image = Image.open(io.BytesIO(image_data))
            images_dict[key] = np.array(image)
            
    labels_dict = None

    # Condition to handle the test data
    if csv_path:
        used_cols = ["isic_id", "target"] #to avoid reading the whole dataframe
        labels_df = pd.read_csv(csv_path, usecols=used_cols)
        labels_dict = dict(zip(labels_df["isic_id"], labels_df["target"]))

    logger.info("Data loded succesfully.")

    return images_dict, labels_dict

def create_transform(image_size, is_train=True):
    '''Create transformations for a given split.
    '''
    logger.info("Creating transformation.")

    # Pretrained params used for efficientNet_b4 during pretraining
    pretrained_size = image_size
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        train_transforms = transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        logger.info("Training transformation created.")
        
        return train_transforms
    
    else:
        valid_transforms = transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        logger.info("Validation/testing transformation created.")
        
        return valid_transforms

def model_fn(model_dir):
    "Function to define the inference/predict call."
    device = "cpu"
    model = net(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))

    model.eval()

    return model

def main(args):
    
    # Instance our model
    model = net(args.device) 

    # Def model configs
    loss_criterion = nn.CrossEntropyLoss() #Contrary to BCEwithLogitsLossL, this won't update the activation function to a sigmoid
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Load data
    train_img_path, train_labels_path = find_files(args.train_dir)
    test_img_path, test_labels_path = find_files(args.test_dir)

    train_images, train_labels = load_data(train_img_path, train_labels_path)
    test_images, test_labels = load_data(test_img_path, test_labels_path)

    # Define transformations
    train_transform = create_transform(args.image_size)
    test_transform = create_transform(args.image_size, is_train=False)

    # Create datasets
    train_dataset = CustomDataset(train_images, train_labels, train_transform)
    test_dataset = CustomDataset(test_images, test_labels, test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    train(model, train_loader, loss_criterion, optimizer, args.epochs, args.device)
    
    # Test the model to see its accuracy
    test(model, test_loader, args.device)
    
    # Save the model
    model_path = os.path.join(args.model_path, "model.pth")
    torch.save(model.cpu().state_dict(), model_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    # Container environment vars.
    parser.add_argument("--model_path", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--image_size", type=int, default=380) # specific to our cancer image data

    args, _ = parser.parse_known_args()

    main(args)
