'''This scrip serves as entrypoint to perform Hyperparameter Optimization using SageMaker's Hyperprameter Tuner.
Here we create an instance of EfficientNet_b4 and define the trainig and testing loop as well as data loaders from S3.'''
import logging
import io
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

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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
    Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Testing started.")
    # Set model to test mode
    model.eval()
    test_loss = 0
    correct = 0

    # With model set to test mode, iterate through data
    with torch.no_grad():
        # loop through data
        for data in test_loader:
            # Move data to device
            data = data.to(device)
            # Make predictions for the test data
            preds = model(data)
            _, prediction = torch.max(preds, 1) # Extract indices/labels
            # Convert to numpy
            prediction = prediction.cpu().numpy() # Leave as array if further operations are needed
    
    # Log predictions
    logger.info(f"Predicted classes for the test data: {prediction.tolist()}")

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
        logger.info(f"Epoch: {epoch}/{epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}")

        # Perform validation
        logger.info("Validation loop started")
        # Set model to evaluation mode
        model.eval()
        val_loss = 0
        correct_val, total_val = 0, 0

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

                # Calculate accuracy
                _, prediction = torch.max(preds, 1) # Get the index of the max log, which corresponds to the label
                correct_val += (prediction == target).sum().item()
                total_val += target.size(0)

            # Log validation metrics
            val_loss /= len(val_loader.dataset)
            val_accuracy = correct_val / total_val
            logger.info(f"Epoch {epoch}/{epochs}, Validation Accuracy: {val_accuracy:.2f}")
            logger.info(f"Validation Loss: {val_loss:.4f}")

    logger.info(f"Finished training for {epochs}s...")


def net(num_classes, device):
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


def main(args):
    
    # Instance our model
    model = net(args.num_classes, args.device) 

    # Def model configs
    loss_criterion = nn.CrossEntropyLoss() #This will update the activation function to a sigmoid activation
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # load data
    train_img_path, train_labels_path = find_files(args.train_dir)
    val_img_path, val_labels_path = find_files(args.val_dir)
    test_img_path, _ = find_files(args.test_dir)

    train_images, train_labels = load_data(train_img_path, train_labels_path)
    val_images, val_labels = load_data(val_img_path, val_labels_path)
    test_images, _ = load_data(test_img_path)

    # Define transformations
    train_transform = create_transform(args.image_size)
    val_transfrom = create_transform(args.image_size, is_train=False)
    test_transfrom = create_transform(args.image_size, is_train=False)

    train_dataset = CustomDataset(train_images, train_labels, train_transform)
    val_dataset = CustomDataset(val_images, val_labels, val_transfrom)
    test_dataset = CustomDataset(test_images, transform=test_transfrom)
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    train(model, train_loader, val_loader, loss_criterion, optimizer, args.epochs, args.device)
    
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
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--image_size", type=int, default=380) # specific to our cancer image data

    args, _ = parser.parse_known_args()

    main(args)
