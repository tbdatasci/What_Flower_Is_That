# Imports here
import sys
#% matplotlib inline
#% config InlineBackend.figure_format = 'retina'

import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim import lr_scheduler

import helper
from PIL import Image
import time
import copy
import seaborn as seaborn


script = sys.argv[0]
data_dir = sys.argv[1]
checkpoint = sys.argv[2]

gpu = False

for i in range(3,len(sys.argv),2):
    
    if sys.argv[i] == '--top_k':
        num_k = int(sys.argv[i+1])
    elif sys.argv[i] == '--category_names':
        cat_names = sys.argv[i+1]
    elif sys.argv[i] == '--gpu':
            gpu = True
    else:
        print("You have elected not to use all of the inputs available")

print("This is my script - ",script, type(script))
print("This is my data directory - ",data_dir, type(data_dir))
print("This is the checkpoint used - ",checkpoint, type(checkpoint))
print("This is the number of top predictions - ",num_k, type(num_k))
print("This is the location of the category map - ",cat_names, type(cat_names))
print("Do we use GPU? ",gpu, type(gpu))

# Load label mapping

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)
    
# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    if chpt['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif chpt['arch'] != 'vgg16':
        print("Base architecture not recognized")
#        break 
    
    model.class_to_idx = chpt['class_to_idx']
    
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, 3136)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(3136, 392)),
                                ('relu', nn.ReLU()),
                                ('fc3', nn.Linear(392, 102)),
                                ('relu', nn.ReLU()),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
        
    # Put the classifier on the pretrained network
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    return model

# Run function and return model

model = load_model(checkpoint)

# Set model to use GPU if GPU was selected
if gpu == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# Define function to process input image

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    
    from PIL import Image
    img = Image.open(image_path)
    
    # Resize the image so the shortest side is 256
    
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
        
    # Create margins, then cut out the center of the picture
    
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    # Scale 0 to 1
    
    img = np.array(img)/255
    
    # Normalize image arrays
    
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Change order of columns due to PyTorch's expectations
    
    img = img.transpose((2, 0, 1))
    
    return img

# Process image

#img2 = process_image(data_dir)

# Define prediction function

def predict(image_path, model, top_num=5):
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

# Predict the image

top_probs, top_labels, top_flowers = predict(data_dir, model, num_k)

print(top_probs)
print(top_labels)
print(top_flowers)