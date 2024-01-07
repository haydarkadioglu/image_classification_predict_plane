# %%
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image

# %%
def load_dataset(input_dir):
    data = []
    labels = []

    folders = []

    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        for folder_file in os.listdir(folder_path):  
            folder_file_path = os.path.join(folder_path, folder_file)
            folders.append(folder_file_path)
           


    for category_idx, category in enumerate(folders):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                img_path = os.path.join(category_path, file)
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img = img.resize((15, 15))  
                    img_array = np.array(img)
                    flattened_img = img_array.flatten()
                    if len(flattened_img) == 15 * 15 * 3:  
                        data.append(flattened_img)
                        labels.append(category_idx)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    return np.array(data), np.array(labels)

# %%
input_dir = "C:\\Users\\ahayd\Desktop\\classification\\data"
data, labels = load_dataset(input_dir)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=45)


# %%
classifier = RandomForestClassifier(n_estimators=100, random_state=50)

classifier.fit(x_train, y_train)


# %%
y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")

# %%
"""import pickle

model_filename = "RF_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(classifier, model_file)"""

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

