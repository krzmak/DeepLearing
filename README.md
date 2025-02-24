# Repository for project developed at the Wrocław University of Science and Technology. Krzysztof Mak

Simple project for classification of traffic signs images

## Contents
- [Info](#info)
- [Tech](#tech)


## Info

- In this project, I created and trained simple custom Cnn model using Pytorch library and then compared it to ResNet-18.
- The model can classify 20 traffic signs.
- Custom model achived best acc of 98.374%, while resnet achived 99.187%.
- To train model, use **`model_cnn.py`** or **`model_resnet.py`**.
- To classify a single image, use **`classify_cnn.py`** or **`classify_resnet.py`**

## Tech

- Python 3.11.4
- PyTorch 2.3.1+cu118
- Torchvision 0.18.1+cu118
- NumPy 1.26.4
- Matplotlib 3.8.4
