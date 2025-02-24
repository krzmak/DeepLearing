# Repository for project developed at the Wrocław University of Science and Technology. Krzysztof Mak

Simple project for classification of traffic signs images

## Contents
- [Info](#info)
- [Tech](#tech)


## Info

- In this project I created and trained simple custom Cnn model using Pytorch liblary and then compared it to resnet18
- Model can classify 20 traffic signs
- Cusmtom model got best acc of 98.374% while resnet got 99.187%
- To train model use **`model_cnn.py`** or **`model_resnet.py`**
- To classify single image use **`classify_cnn.py`** or **`classify_resnet.py`**

## Tech

- Python 3.11.4
- PyTorch 2.3.1+cu118
- Torchvision 0.18.1+cu118
- NumPy 1.26.4
- Matplotlib 3.8.4
