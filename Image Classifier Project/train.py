# Imports:
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
import time
import numpy as np
from PIL import Image
import sys
import os
import random
import argparse

# Default variables

arch = "resnet152"
learning_rate = 0.001
hidden_units = 1024
epochs = 10
device = "cpu"

# Parameters with argparse

parser = argparse.ArgumentParser(description = "Train image classifier model")
parser.add_argument("data_dir", type=str, help="Define the directory which contains image data for training")
parser.add_argument("-a", "--arch", action="store", type=str, help="Please choose from the following 3 pretrained models: resnet152, vgg19, densenet161")
parser.add_argument("-lr", "--learning_rate", action="store", type=float, help="Please type in a float number for learning rate")
parser.add_argument("-H", "--hidden_units", action="store", type=int, help="Please type in an integer number to set the number of hidden units in the 1st layer")
parser.add_argument("-e", "--epochs", action="store", type=int, help="Please type in an integer number to set the number of epochs in the training loop")
parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU if it is available")
parser.add_argument("-s", "--save_dir", action="store", type=str, help="Define the directory for saving the trained model")

params = parser.parse_args()

if params.arch:
    arch = params.arch
if params.learning_rate:
    learning_rate = params.learning_rate
if params.hidden_units:
    hidden_units = params.hidden_units
if params.epochs:
    epochs = params.epochs
if params.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define datasets
data_dir = params.data_dir
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
test_dir = data_dir + "/test"

# Define transformations for the datasets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Define datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Define dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
   
    
# Function for initiating the model
def initiate_model(arch="resnet152", hidden_units=512, learning_rate=0.001):
    '''
    Initiate model with function
    '''
    
    model = getattr(models, arch)(pretrained=True)
    if arch == "resnet152":
        inputs = model.fc.in_features
    elif arch == "vgg19":
        inputs = model.classifier[0].in_features
    elif arch == "densenet161":
        inputs = model.classifier.in_features
    else:
        print("Please choose another pretrained model architecture")
        
    # Freeze parameters
    
    for param in model.parameters():
        param.requires_grad = False
        
    # Modify classifier
    classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(inputs, hidden_units)),
                                            ("dropout", nn.Dropout(0.2)),
                                            ("relu", nn.ReLU()),
                                            ("fc2", nn.Linear(hidden_units, 102)),
                                            ("output", nn.LogSoftmax(dim=1))]))
    
    if arch == "resnet152":
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
    criterion = nn.NLLLoss()
    
    return model, criterion, optimizer

model, criterion, optimizer = initiate_model(arch, hidden_units, learning_rate)

print("\nThe model has been initiated")

# Function for training the model
def train_model(model, criterion, optimizer, epochs):
    '''
    This function trains the pretrained model on our dataset
    '''
    model.to(device)
    training_start = time.time()
    print("\nStarting training...")
    
    
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            valid_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    
                    log_ps = model.forward(images)
                    valid_loss += criterion(log_ps, labels)
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
            model.train()
                
            
            print("\nEpochs: {}/{}:\n".format(e+1, epochs),
                  "Train loss: {:.3f}\n".format(running_loss/len(trainloader)),
                  "Valid loss: {:.3f}\n".format(valid_loss/len(validloader)),
                  "Accuracy: {:.3f}\n".format(accuracy/len(validloader)))
            
    training_end = time.time()
    training_time = training_end - training_start
    print("Training time: {} min {} sec".format(int(training_time//60), int(training_time%60)))

    return model


trained_model = train_model(model, criterion, optimizer, epochs)
print("\nThe model has been trained.")


# Save function for the trained model
def save_model(trained_model):
    '''
    This function saves the trained model to the chosen directory
    '''
    
    trained_model.class_to_idx = train_data.class_to_idx
    save_dir = ""
    
    checkpoint = {"input_size": trained_model.fc[0].in_features if arch == "resnet152" else trained_model.classifier[0].in_features,
                  "output_size": 102,
                  "pretrained": arch,
                  "state_dict": trained_model.state_dict(),
                  "epochs": epochs,
                  "optimizer": optimizer.state_dict(),
                  "classifier": trained_model.fc if arch == "resnet152" else trained_model.classifier,
                  "class_to_idx": trained_model.class_to_idx}
    
    if params.save_dir:
        save_dir = params.save_dir
    else:
        save_dir = "checkpoint.pth"
        
    torch.save(checkpoint, save_dir)
    
    
# Save the trained model

save_model(trained_model)
print("The model has been saved.")
