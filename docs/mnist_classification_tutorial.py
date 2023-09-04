from pytorch_ml.train import *
from pytorch.metrics import accuracy

import torchvision as tv
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch
from torch import nn
from torch.utils.data import DataLoader

# Get the Mnist dataset
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

# Define classes
class_names = train_data.classes

# Define DataLoaders
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

# Define Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# Define Model, loss, optimizer
model = MNIST_CNN_classifier(n_classes=len(class_names))
optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters()) # recall parameters are just layers in pytorch with weights
loss_fn = nn.CrossEntropyLoss() # recall that pytorch CrossEntropy requires the targets to be int64 or longtensors

# Train with train_manager
train_manager(train_dataloader=train_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy,
        device=device,
        validation_step=True,
        val_dataloader=test_dataloader,
        epochs=3
    )