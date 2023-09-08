#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 11 Categorizing Images of Clothing with Convolutional Neural Networks 
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Exploring the clothing image dataset 

import torch, torchvision 
from torchvision import transforms 

image_path = './'
transform = transforms.Compose([transforms.ToTensor(),
                               # transforms.Normalize((0.5,), (0.5,))
                               ])

train_dataset = torchvision.datasets.FashionMNIST(root=image_path, 
                                                  train=True, 
                                                  transform=transform, 
                                                  download=True)

test_dataset = torchvision.datasets.FashionMNIST(root=image_path, 
                                                 train=False, 
                                                 transform=transform, 
                                                 download=False)


print(train_dataset)


print(test_dataset)


from torch.utils.data import DataLoader

batch_size = 64
torch.manual_seed(42)
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)


data_iter = iter(train_dl)
images, labels = next(data_iter)


print(labels)


# constant for classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


print(images[0].shape)


print(torch.max(images), torch.min(images))


import numpy as np
import matplotlib.pyplot as plt

npimg = images[1].numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.colorbar()
plt.title(class_names[labels[1]])
plt.show()


plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.subplots_adjust(hspace=.3)
    plt.xticks([])
    plt.yticks([])
    npimg = images[i].numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="Greys")
    plt.title(class_names[labels[i]])
plt.show()


# # Classifying clothing images with CNNs 

# ## Architecting the CNN model 

import torch.nn as nn
model = nn.Sequential()


model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3))
model.add_module('relu1', nn.ReLU()) 


model.add_module('pool1', nn.MaxPool2d(kernel_size=2))


model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
model.add_module('relu2', nn.ReLU())   


model.add_module('pool2', nn.MaxPool2d(kernel_size=2))


model.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
model.add_module('relu3', nn.ReLU()) 


x = torch.rand((64, 1, 28, 28))
print(model(x).shape)


model.add_module('flatten', nn.Flatten()) 


print(model(x).shape)


model.add_module('fc1', nn.Linear(1152, 64))
model.add_module('relu4', nn.ReLU()) 


model.add_module('fc2', nn.Linear(64, 10))
model.add_module('output', nn.Softmax(dim = 1))


print(model)


from torchsummary import summary


summary(model, input_size=(1, 28, 28), batch_size=-1, device="cpu")


# ## Fitting the CNN model 

device = torch.device("cuda:0")
# device = torch.device("cpu")
model = model.to(device) 

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, optimizer, num_epochs, train_dl):
    for epoch in range(num_epochs):
        loss_train = 0
        accuracy_train = 0
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device) 
            y_batch = y_batch.to(device) 
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_train += is_correct.sum().cpu()

        loss_train /= len(train_dl.dataset)
        accuracy_train /= len(train_dl.dataset)
        
        print(f'Epoch {epoch+1} - loss: {loss_train:.4f} - accuracy: {accuracy_train:.4f}')


num_epochs = 30
train(model, optimizer, num_epochs, train_dl)


test_dl = DataLoader(test_dataset, batch_size, shuffle=False)

def evaluate_model(model, test_dl):
    accuracy_test = 0
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            pred = model.cpu()(x_batch)
            is_correct = torch.argmax(pred, dim=1) == y_batch
            accuracy_test += is_correct.float().sum().item()
    
    print(f'Accuracy on test set: {100 * accuracy_test / 10000} %')

evaluate_model(model, test_dl)


# ## Visualizing the convolutional filters 

conv3_weight = model.conv3.weight.data
print(conv3_weight.shape)


plt.figure(figsize=(10, 10))

n_filters = 16
for i in range(n_filters):
    weight = conv3_weight[i].cpu().numpy()
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(weight[0], cmap='gray')
 
plt.show()


# # Boosting the CNN classifier with data augmentation 

# ## Flipping for data augmentation

def display_image_greys(image):
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="Greys")
    plt.xticks([])
    plt.yticks([])


image = images[1]
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
display_image_greys(image)

## flipping (horizontally)
img_flipped = transforms.functional.hflip(image)
plt.subplot(1, 2, 2)
display_image_greys(img_flipped)

plt.show()
 


torch.manual_seed(42)
flip_transform = transforms.Compose([transforms.RandomHorizontalFlip()])

plt.figure(figsize=(10, 10))
plt.subplot(1, 4, 1)
display_image_greys(image)

for i in range(3):
    plt.subplot(1, 4, i+2)
    img_flip = flip_transform(image)
    display_image_greys(img_flip)


# ## Rotation for data augmentation

# rotate

torch.manual_seed(42)
rotate_transform = transforms.Compose([transforms.RandomRotation(20)])

plt.figure(figsize=(10, 10))
plt.subplot(1, 4, 1)
display_image_greys(image)

for i in range(3):
    plt.subplot(1, 4, i+2)
    img_rotate = rotate_transform(image)
    display_image_greys(img_rotate)


# ## Cropping for data augmentation

torch.manual_seed(42)
crop_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.7, 1))])

plt.figure(figsize=(10, 10))
plt.subplot(1, 4, 1)
display_image_greys(image)

for i in range(3):
    plt.subplot(1, 4, i+2)
    img_crop = crop_transform(image)
    display_image_greys(img_crop)
    


# # Improving the clothing image classifier with data augmentation

torch.manual_seed(42)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1)),
    transforms.ToTensor(),
])

 
train_dataset_aug = torchvision.datasets.FashionMNIST(root=image_path, 
                                                      train=True, 
                                                      transform=transform_train, 
                                                      download=False)


from torch.utils.data import Subset
train_dataset_aug_small = Subset(train_dataset_aug, torch.arange(500)) 


train_dl_aug_small = DataLoader(train_dataset_aug_small, batch_size, shuffle=True)


model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3))
model.add_module('relu1', nn.ReLU()) 
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))

model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
model.add_module('relu2', nn.ReLU())   
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))

model.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
model.add_module('relu3', nn.ReLU()) 

model.add_module('flatten', nn.Flatten()) 
model.add_module('fc1', nn.Linear(1152, 64))
model.add_module('relu4', nn.ReLU()) 

model.add_module('fc2', nn.Linear(64, 10))
model.add_module('output', nn.Softmax(dim = 1))

model = model.to(device) 


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)        
train(model, optimizer, 1000, train_dl_aug_small)


evaluate_model(model, test_dl)


# # Advancing the CNN classifier with transfer learning  

from torchvision.models import resnet18
my_resnet = resnet18(weights='IMAGENET1K_V1')


my_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = my_resnet.fc.in_features
my_resnet.fc = nn.Linear(num_ftrs, 10)


my_resnet = my_resnet.to(device) 
optimizer = torch.optim.Adam(my_resnet.parameters(), lr=0.001)  
train(my_resnet, optimizer, 10, train_dl)


evaluate_model(my_resnet, test_dl)


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch11_part1.ipynb --TemplateExporter.exclude_input_prompt=True')

