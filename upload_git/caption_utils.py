import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import os

# Load the model and tokenizer
def load_model():
    model_path = '/home/abdelraheem/Abdo_Omda/Nlp/model_weights/final_model.h5' 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = tf.keras.models.load_model(model_path)
    
    # Load the tokenizer from a file or initialize it
    # Assuming tokenizer was saved using pickle or json
    tokenizer_path = '/path/to/tokenizer.pkl'  # Update this with the correct path to your tokenizer
    tokenizer = None
    if os.path.exists(tokenizer_path):
        import pickle
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)  # Load tokenizer from file
    
    return model, tokenizer

# Preprocess the image (resize and normalize)
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((299, 299))  # Resize based on model's input size (e.g., 299x299 for InceptionV3)
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Generate caption from model
def generate_caption(model, tokenizer, image_path):
    img = preprocess_image(image_path)
    
    # Predict the caption (this depends on your model's architecture)
    # Assuming a simple model that directly predicts the caption from the image features
    features = model.predict(img)  # This is just an example; replace it with your actual model logic

    # Convert model output to human-readable caption using the tokenizer
    # Assuming features are image embeddings and tokenizer is used for caption generation
    # Decode the prediction into words
    caption = decode_prediction(features, tokenizer)
    return caption

# Decode model prediction into a readable caption
def decode_prediction(features, tokenizer):
    # This part depends on your model output. If your model outputs indices, we need to convert them to words.
    # Here's an example using a sequence-based model:
    
    # For demonstration, assuming the model gives word indices as output
    predicted_sequence = features.argmax(axis=-1)  # Get the predicted word indices
    caption = []
    
    for idx in predicted_sequence[0]:
        word = tokenizer.index_word.get(idx, '<unk>')  # Get the word from the index, or use <unk> for unknown words
        if word == '<end>':  # Stop when the end token is encountered
            break
        caption.append(word)
    
    return ' '.join(caption)

# Function to run the full caption generation pipeline
def run_caption_generation(image_path):
    model, tokenizer = load_model()
    caption = generate_caption(model, tokenizer, image_path)
    return caption
