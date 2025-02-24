import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image

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

model = torch.load('model.pth')

mean = [0.5024, 0.4449, 0.4261]
std = [0.2323, 0.2307, 0.2292]

image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)

    _, predicted = torch.max(output.data, 1)

    print(classes[predicted.item()])

classify(model, image_transforms, "Val/B36/17.jpg", classes)