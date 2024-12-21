import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Path to your dataset
data_dir =r"C:\Users\nandhini\OneDrive\Desktop\SALT_PROJECT\DATASET"

# Loading the data
def load_data():
    images = []
    labels = []
    categories = ["Pure salt", "Algae", "Dust", "Metals", "Sand", "Stone"]
    for category in categories:
        folder_path = os.path.join(data_dir, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            # Load and preprocess the image
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(category)
    return np.array(images), np.array(labels)

# Load the dataset
images, labels = load_data()

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

# Define the model architecture
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Build and train the model
model = build_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save the trained model
model.save(r"C:\Users\nandhini\OneDrive\Desktop\SALT_PROJECT\salt_impurity_model.h5")
