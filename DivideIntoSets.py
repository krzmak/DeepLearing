import os
import shutil
from sklearn.model_selection import train_test_split

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

base_dir = 'Data'
train_dir = 'Train'
val_dir = 'Val'
test_dir = 'Test'

for dir in [train_dir, val_dir, test_dir]:
    create_dir_if_not_exists(dir)

def split_data(base_dir, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
            
            train_images, temp_images = train_test_split(images, train_size=train_size, random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=test_size/(test_size + val_size), random_state=42)
            
            for image in train_images:
                dest_dir = os.path.join(train_dir, class_folder)
                create_dir_if_not_exists(dest_dir)
                shutil.copy(image, dest_dir)
            
            for image in val_images:
                dest_dir = os.path.join(val_dir, class_folder)
                create_dir_if_not_exists(dest_dir)
                shutil.copy(image, dest_dir)
            
            for image in test_images:
                dest_dir = os.path.join(test_dir, class_folder)
                create_dir_if_not_exists(dest_dir)
                shutil.copy(image, dest_dir)

split_data(base_dir, train_dir, val_dir, test_dir)
