import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn as nn
import torch.nn.functional as F

# Definicja modelu CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Lista klas
classes = [
    'A1 - niebezpieczny zakręt w prawo', 
    'A2 - niebezpieczny zakręt w lewo', 
    'A7 - ustąp pierwszeństwa', 
    'A17 - dzieci', 
    'A21 - tramwaj', 
    'A30 - inne niebezpieczeństwo', 
    'B1 - zakaz ruchu w obu kierunkach', 
    'B2 - zakaz wjazdu', 
    'B20 - stop', 
    'B21 - zakaz skręcania w lewo', 
    'B22 - zakaz skręcania w prawo',
    'B23 - zakaz zawracania', 
    'B33 - ograniczenie prędkości', 
    'B36 - zakaz zatrzymywania się',
    'B41 - zakaz ruchu pieszych',  
    'C2 - nakaz jazdy w prawo za znakiem', 
    'C4 - nakaz jazdy w lewo za znakiem', 
    'C12 - ruch okrężny', 
    'D1 - droga z pierwszeństwem', 
    'D6 - przejście dla pieszych'
]

# Średnia i odchylenie standardowe dla normalizacji
mean = [0.5024, 0.4449, 0.4261]
std = [0.2323, 0.2307, 0.2292]

# Transformacje obrazu
image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model, image_transforms, image_path, classes):
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
    
    _, predicted = torch.max(output.data, 1)
    print(f"Predykcja: {classes[predicted.item()]}")
    
    for idx, class_name in enumerate(classes):
        print(f"{class_name}: {probabilities[0][idx].item() * 100:.2f}%")

cnn_model = SimpleCNN(num_classes=20)
checkpoint = torch.load('best_model_cnn_checkpoint.pth.tar')
cnn_model.load_state_dict(checkpoint['model'])
cnn_model = cnn_model.eval()

classify(cnn_model, image_transforms, "Val/B36/17.jpg", classes)
