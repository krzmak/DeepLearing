import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('Train', transform=data_transforms)
val_dataset = datasets.ImageFolder('Val', transform=data_transforms)
test_dataset = datasets.ImageFolder('Test', transform=data_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Dictonary

# 'A1 - niebezpieczny zakręt w prawo', 
# 'A2 - niebezpieczny zakręt w lewo', 
# 'A7 - ustąp pierwszeństwa', 
# 'A17 - dzieci', 
# 'A21 - tramwaj', 
# 'A30 - inne niebezpieczeństwo', 
# 'B1 - zakaz ruchu w obu kierunkach', 
# 'B2 - zakaz wjazdu', 
# 'B20 - stop', 
# 'B21 - zakaz skręcania w lewo', 
# 'B22 - zakaz skręcania w prawo',
# 'B23 - zakaz zawracania', 
# 'B33 - ograniczenie prędkości', 
# 'B36 - zakaz zatrzymywania się',
# 'B41 - zakaz ruchu pieszych',  
# 'C2 - nakaz jazdy w prawo za znakiem', 
# 'C4 - nakaz jazdy w lewo za znakiem', 
# 'C12 - ruch okrężny', 
# 'D1 - droga z pierwszeństwem', 
# 'D6 - przejście dla pieszych'

def get_mean_and_std(loader):
    mean = 0.
    std =0.
    total_image_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_image_count += image_count_in_a_batch

    mean /= total_image_count
    std /= total_image_count

    return mean, std    

print(get_mean_and_std(train_loader))