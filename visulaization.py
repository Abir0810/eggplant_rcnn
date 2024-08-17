import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from tensorflow.keras.applications import MobileNetV2

# Define paths to the dataset
train_dir = r'E:\AI\Eggplant\Eggplant Disease Recognition Dataset\Augmented Images\train'
val_dir = r'E:\AI\Eggplant\Eggplant Disease Recognition Dataset\Augmented Images\val'

# Load the data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Get class labels
class_names = list(train_generator.class_indices.keys())

# Visualize sample images from each class
def visualize_samples(generator, class_names):
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        data, _ = next(generator)
        image = data[i]
        plt.subplot(2, len(class_names)//2 + 1, i + 1)
        plt.imshow(image)
        plt.title(class_name)
        plt.axis('off')
    plt.show()

visualize_samples(train_generator, class_names)

# Get the count of each class in the training dataset
class_counts = {class_name: len(os.listdir(os.path.join(train_dir, class_name))) for class_name in class_names}

# Plot the class distribution
plt.figure(figsize=(15, 6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
plt.title('Class Distribution in Training Data')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.show()