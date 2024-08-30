import os
import sys
import logging

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net(device):
    # Load the pretrained model
    model = models.resnet50(pretrained=True)
    logger.info("Loaded pretrained ResNet50.")

    # Freeze the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the FC layer
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, 2)

    model.to(device)
    logger.info("Pretrained model created.")

    return model

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instance pretrained net
    model = net(device)
    logger.info(f"Pretrained model instance in {device}.")

    # Load our trained model
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))

    # Set model to evaluation mode
    model.eval()
    logger.info("Succesfully loaded model in inference mode.")

    return model
