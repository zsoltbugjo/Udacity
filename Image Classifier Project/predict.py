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
import pandas as pd

# Random image
folder = str(random.randint(1, 102))
folder_path = "flowers/test/" + folder + "/"
random_img = folder_path + random.choice(os.listdir(folder_path))

# Default variables

checkpoint = "checkpoint.pth"
image_path = random_img
category_names = "cat_to_name.json"
topk = 5

# Parameters with argparse

parser = argparse.ArgumentParser(description="Predict the category of a chosen or a random image")
parser.add_argument("-c", "--checkpoint", action="store", type=str, help="Name of the model that will be used for prediction")
parser.add_argument("-i", "--image_path", action="store", type=str, help="Path to the image to be predicted")
parser.add_argument("-n", "--category_names", action="store", type=str, help="The name of the json file which contains the categories")
parser.add_argument("-t", "--topk", action="store", type=int, help="The number of categories to display")
parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU if it is available")

params = parser.parse_args()

if params.checkpoint:
    checkpoint = params.checkpoint
if params.image_path:
    image_path = params.checkpoint
if params.category_names:
    category_names = params.category_names
if params.topk:
    topk = params.topk
if params.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
with open(category_names, "r") as f:
    cat_to_name = json.load(f)


    
# load trained model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint["pretrained"])(pretrained=True)
    if checkpoint["pretrained"] == "resnet152":
        model.fc = checkpoint["classifier"]
    else:
        model.classifier = checkpoint["classifier"]
    model.epochs = checkpoint["epochs"]
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    return model


model = load_checkpoint(checkpoint)

# Image processing
def process_image(image):
    '''
    Scales, crops and normalizes a PIL image for a PyTorch model, returns a numpy array
    '''
    
    im = Image.open(image)
    im.thumbnail((256, 256))
    dist = (256-224)/2
    im = im.crop((dist, dist, 256-dist, 256-dist))
    np_image = np.array(im)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (np_image - mean) / std
    
    transposed = im.transpose(2, 0, 1)
    return transposed

# function for getting the labels
def get_label(ind):
    for label, index in model.class_to_idx.items():
        if ind == index:
            return label
    return "key doesn't exist"

# prediction function
def predict(image_path, model, topk=5):
    '''
    Predict the class (or classes) of an image using a trained deep learning model
    '''
    model.to(device)
    model.eval()
    
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)
    
    log_ps = model.forward(image)
    ps = torch.exp(log_ps)
    
    probs = ps.topk(topk)[0].tolist()[0]
    indexes = ps.topk(topk)[1].tolist()[0]
    
    labels = []
    for i in range(len(indexes)):
        label = get_label(indexes[i])
        labels.append(label)
        
    return probs, labels


probs, labels = predict(random_img, model)
category = cat_to_name[folder]
classes = [cat_to_name[i] for i in labels]
predictions = pd.DataFrame(list(zip(probs, classes)), columns=["probabilities", "classes"])

print("Chosen category: {}\n".format(category),
      "The top {} most likely classes are the following:\n".format(topk),
      predictions)

