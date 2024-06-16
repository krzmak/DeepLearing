import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = [0.5024, 0.4449, 0.4261]
std = [0.2323, 0.2307, 0.2292]

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

train_dataset = datasets.ImageFolder('Train', transform=data_transforms)
val_dataset = datasets.ImageFolder('Val', transform=data_transforms)
test_dataset = datasets.ImageFolder('Test', transform=data_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow = 3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()
    print('labels:' , labels)

show_transformed_images(train_dataset)

resnet18_model = models.resnet18(weights=None)
num_of_features = resnet18_model.fc.in_features
number_of_classes = 20
resnet18_model.fc = nn.Linear(num_of_features, number_of_classes)
resnet18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

def train_nn(model = resnet18_model, train_loader = train_loader, test_loader = test_loader, crtiterion = loss_fn, optimiser = optimizer, n_epochs = 100):
    best_acc = 0

    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimiser.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = crtiterion(outputs, labels)
            
            loss.backward()

            optimiser.step()

            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100 * running_correct/ total

        print("         -Trainig dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f" % (running_correct, total, epoch_acc, epoch_loss))

        test_dataset_acc = evaluate_model_on_test_set(model, test_loader)

        if (test_dataset_acc > best_acc):
            best_acc = test_dataset_acc
            save_checkpoint(model, epoch , optimiser, best_acc)

    print("Finished")
    return model


def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100.0 *predicted_correctly_on_epoch / total

    print("         -Testing dataset. Got %d of %d images correctly ((%.3f%%)" % (predicted_correctly_on_epoch, total, epoch_acc))

    return epoch_acc

def save_checkpoint(model, epoch , optimiser, best_acc):
    state = {
        'epoch' : epoch +1,
        'model' : model.state_dict(),
        'best accuracy' : best_acc,
        'optimizer' : optimiser.state_dict(),
    }
    torch.save(state, 'best_model_checkpoint.pth.tar')

train_nn(model = resnet18_model, train_loader = train_loader, test_loader = test_loader, crtiterion = loss_fn, optimiser = optimizer, n_epochs = 100)


checkpoint = torch.load('best_model_checkpoint.pth.tar')

resnet18_model = models.resnet18(weights=None)
num_of_features = resnet18_model.fc.in_features
number_of_classes = 20
resnet18_model.fc = nn.Linear(num_of_features, number_of_classes)
resnet18_model.load_state_dict(checkpoint['model'])

torch.save(resnet18_model, 'model.pth')

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