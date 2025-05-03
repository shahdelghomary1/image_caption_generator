import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os

# Function to load the model and related components
def setup_pipeline():
    model_path = '/home/abdelraheem/Abdo_Omda/Nlp/model_weights/final_model.h5'
    tokenizer_path = '/home/abdelraheem/Abdo_Omda/Nlp/tokenizer.pkl' 
    max_length = 34  
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the feature extraction model (for InceptionV3 or similar)
    feature_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    wordtoix = tokenizer.word_index 
    ixtoword = {i: word for word, i in wordtoix.items()} 
    
    return model, feature_model, wordtoix, ixtoword, max_length

# Function to encode the image for feature extraction
def encode_image(img_path, feature_model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    
    features = feature_model.predict(img_array)
    return features

def generate_caption(model, test_feat, wordtoix, ixtoword, max_length):
    seq = [1]  
    
    for _ in range(max_length):
        sequence = pad_sequences([seq], maxlen=max_length)
        
        yhat = model.predict([test_feat, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        seq.append(yhat)
        
        if yhat == wordtoix['<end>']:
            break
    
    caption = ' '.join([ixtoword[idx] for idx in seq[1:]])  # Exclude <start> token
    return caption

def run_caption_generation(test_img_path):
    model, feature_model, wordtoix, ixtoword, max_length = setup_pipeline()
    test_feat = encode_image(test_img_path, feature_model).squeeze()
    caption = generate_caption(model, np.expand_dims(test_feat, axis=0), wordtoix, ixtoword, max_length)
    
    return caption

def app():
    st.title("🖼️ Image Caption Generator")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        if st.button("Generate Caption"):
            temp_img_path = "temp_uploaded_img.jpg"
            image.save(temp_img_path)

            caption = run_caption_generation(temp_img_path)

            st.markdown("**📝 Generated Caption:**")
            st.success(caption)

            st.image(temp_img_path, caption="Uploaded Image", use_container_width=True)
            
if __name__ == "__main__":
    app()
