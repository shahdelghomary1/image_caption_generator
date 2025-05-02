import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="nlp",
    initial_sidebar_state="expanded"
)

import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import os
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, concatenate
from tensorflow.keras.models import Model
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import os
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, concatenate
from tensorflow.keras.models import Model

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
    }
    .stButton>button {
        background: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .caption-box {
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-top: 1rem;
        border: 2px solid #4CAF50;
    }
    .arabic-text {
        font-family: 'Arial', sans-serif;
        font-size: 1.2rem;
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- Model Loading --------------------
@st.cache_resource
def load_components():
    feature_model = DenseNet201(
        include_top=False, 
        weights='imagenet', 
        pooling='avg', 
        input_shape=(224, 224, 3)
    )
    
    try:
        with open("C:\\Users\\Lenovo\\OneDrive - Egyptian E-Learning University\\Documents\\shahd\\tokenizer.pkl", 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"🔑 Tokenizer Error: {str(e)}")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(
            "C:\\Users\\Lenovo\\OneDrive - Egyptian E-Learning University\\Documents\\shahd\\ImageCaptioningModels\\model1_lstm_full_model.keras"
        )
    except Exception as e:
        st.error(f"🧠 Model Error: {str(e)}")
        st.stop()
    
    return feature_model, tokenizer, model

feature_model, tokenizer, model = load_components()

# -------------------- Constants --------------------
MAX_LENGTH = 37
VOCAB_SIZE = len(tokenizer.word_index) + 1

# -------------------- Helper Functions --------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.densenet.preprocess_input(img_array)

@st.cache_data(max_entries=3)
def get_image_features(img):
    processed_img = preprocess_image(img)
    return feature_model.predict(processed_img, verbose=0).squeeze()

def generate_caption(features):
    caption = 'startseq'
    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=MAX_LENGTH, padding='post')
        pred = model.predict_on_batch([features.reshape(1, -1), seq])
        pred_word = np.argmax(pred)
        
        word = next(
            (w for w, i in tokenizer.word_index.items() if i == pred_word),
            None
        )
        
        if not word or word == 'endseq':
            break
            
        caption += ' ' + word
    
    return caption.replace('startseq ', '').replace(' endseq', '')

# -------------------- UI --------------------
# Replace UI elements with English-only content
st.title("Image Caption Generator")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Maximum file size: 4MB"
)

if uploaded_file:
    col1, col2 = st.columns([2, 3], gap="large")
    
    with col1:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(
            img, 
            caption="Uploaded Image",
            use_column_width=True,
            output_format="auto"
        )
    
    with col2:
        st.subheader("Generated Caption")
        with st.spinner('Generating...'):
            try:
                features = get_image_features(img)
                caption = generate_caption(features)
                
                st.markdown(f"""
                <div class="caption-box">
                    <p style="font-size: 1.2rem; margin: 0;">{caption.capitalize()}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Caption generation failed: {str(e)}")

st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #a0a0a0;">
        Powered by Shahd ElGhomay
    </div>
""", unsafe_allow_html=True)
