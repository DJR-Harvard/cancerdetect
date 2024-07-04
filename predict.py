import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('breast_cancer_model.h5')

# Function to predict cancer from a single image
def predict_cancer(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    prediction = model.predict(img_array)
    return prediction[0][0]

# Example usage
img_path = '/Users/davidrasmussen/cancerdetect/dataset/15516/1/15516_idx5_x1301_y1751_class1.png'  # Replace with the actual path to your image
probability = predict_cancer(img_path)
print(f'Probability of breast cancer: {probability:.2f}')
