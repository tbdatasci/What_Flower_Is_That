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

gpu = False

for i in range(2,len(sys.argv),2):
    
    if sys.argv[i] == '--save_dir':
        save_dir = sys.argv[i+1]
    elif sys.argv[i] == '--arch':
        arch = sys.argv[i+1]
    elif sys.argv[i] == '--learning_rate':
        learning_rate = float(sys.argv[i+1])
    elif sys.argv[i] == '--hidden_units':
        hidden_units = int(sys.argv[i+1])
    elif sys.argv[i] == '--epochs':
        epochs = int(sys.argv[i+1])
    elif sys.argv[i] == '--gpu':
            gpu = True
    else:
        print("You have elected not to use all of the inputs available")

print("This is my script - ",script, type(script))
print("This is my data directory - ",data_dir, type(data_dir))
print("This is where I'll save my checkpoint - ",save_dir, type(save_dir))
print("This is my model architecture - ",arch, type(arch))
print("This is my learning rate - ",learning_rate, type(learning_rate))
print("These are my hidden units - ",hidden_units, type(hidden_units))
print("These is my number of epochs - ",epochs, type(epochs))
print("Do we use GPU? ",gpu, type(gpu))
            
# Define the directory and assign train, validation, and test folders
#data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

dirs = {'train' : train_dir,
        'valid' : valid_dir,
        'test' : test_dir}

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(dirs[x],
transform = data_transforms[x]) for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
batch_size = 32, shuffle = True) for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x])
                for x in ['train', 'valid', 'test']}

class_names = image_datasets['train'].classes

# Set model to use GPU if GPU was selected
if gpu == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# Add label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Import chosen model

if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif arch == 'vgg19':
    model = models.vgg19(pretrained=True)
    
# Define classifier and append it to chosen model
classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, 3136)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(3136, 392)),
                                ('relu', nn.ReLU()),
                                ('fc3', nn.Linear(392, 102)),
                                ('relu', nn.ReLU()),
                                ('output', nn.LogSoftmax(dim=1))
]))

for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier

# Tell PyTorch to use cuda if available

model = model.to(device)

# Define how to train the model

def train_model(model, criteria, optimizer, scheduler,    
                                      num_epochs=10, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs =inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criteria(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Define model inputs

criteria = nn.NLLLoss()

# Observe that all parameters are being optimized
optim = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Decay LR by a factor of 0.1 every 4 epochs
sched = lr_scheduler.StepLR(optim, step_size=4, gamma=0.1)

# Number of epochs
eps=epochs


# Train the model

model_trained = train_model(model, criteria, optim, sched, eps, 'cuda')


# TODO: Do validation on the test set


# Define accuracy evaluation

def calc_accuracy(model, data, cuda=False):
    model.eval()
    model.to(device='cuda')    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # get model outputs
            outputs = model.forward(inputs)
            # convert outputs to predictions
            _, predicted = outputs.max(dim=1)
            
            if idx == 0:
                print(predicted) # class
                print(torch.exp(_)) # probability
            equals = predicted == labels.data
            if idx == 0:
                print(equals)
            print(equals.float().mean())
            
# Check accuracy on validation set
calc_accuracy(model, 'test', True)

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'arch': arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)