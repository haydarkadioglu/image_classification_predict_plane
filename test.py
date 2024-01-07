from PIL import Image
import numpy as np
import pickle
import os

with open("RF_model.pkl", 'rb') as model_file:
    loaded_model = pickle.load(model_file)


folders = []
input_dir = "C:\\Users\\ahayd\\Desktop\\classification\\data"
for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)
    for folder_file in os.listdir(folder_path):  
        folder_file_path = os.path.join(folder_path, folder_file)
        folders.append(folder_file_path.split('\\')[-1])
        

def predict_image_class(image_path, model):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((15, 15))  
    img_array = np.array(img)
    flattened_img = img_array.flatten()
    
    predicted_class = model.predict([flattened_img])[0]
    
    return predicted_class


image_path = "C:\\Users\\ahayd\\Desktop\\classification\\test_images\\A380.jpg"
predicted_class = predict_image_class(image_path, loaded_model)

print(f"Tahmin edilen class: {folders[predicted_class]}")