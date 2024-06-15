from PIL import Image
import os

def resize_images(directory, size=(128, 128)):
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if file_path.endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(file_path)
                    img = img.resize(size, Image.LANCZOS)
                    img.save(file_path)

resize_images('Data')
