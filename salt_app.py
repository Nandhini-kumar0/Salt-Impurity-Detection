from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Load the trained model
model = load_model(r"C:\Users\nandhini\OneDrive\Desktop\SALT_PROJECT\salt_impurity_model.h5")

# Label Encoder (make sure it matches the one used during training)
categories = ["Pure salt", "Algae", "Dust", "Metals", "Sand", "Stone"]
label_encoder = {i: category for i, category in enumerate(categories)}

# Initialize Flask app
app = Flask(__name__)

# Define the route to upload images
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image
        img_file = request.files['image']
        img_path = os.path.join(r"C:\Users\nandhini\OneDrive\Desktop\SALT_PROJECT\static\uploads", img_file.filename)
        img_file.save(img_path)

        # Preprocess the image for prediction
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the impurities
        predictions = model.predict(img_array)[0]
        impurity_percentages = {label_encoder[i]: round(pred * 100, 2) for i, pred in enumerate(predictions)}

        return render_template('index.html', impurity_percentages=impurity_percentages, image_path=img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
