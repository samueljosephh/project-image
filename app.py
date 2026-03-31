import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os
from gtts import gTTS
import io

st.set_page_config(page_title="VisionTalk", page_icon="👁️")

st.title("VisionTalk: Image Captioning Generator")
st.markdown("""
### 👁️ See the World Through AI
This platform is designed to aid visually impaired users by describing the world around them.
Upload an image and our Multimodal AI (CNN + LSTM) will generate a caption and speak it aloud.
""")

# Load models
@st.cache_resource
def load_captioning_model():
    model_path = 'best_model.keras'
    tokenizer_path = 'tokenizer.pkl'
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        try:
            model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    return None, None

@st.cache_resource
def load_vgg_model():
    # We use the layer before the final classification to get the 4096-dim vector
    base_model = VGG16()
    vgg_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return vgg_model

model, tokenizer = load_captioning_model()
vgg_model = load_vgg_model()

# Helper function to map index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Improved Prediction Function
def predict_caption(model, image, tokenizer, max_length=34):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict word probabilities
        yhat = model.predict([image, sequence], verbose=0)
        
        # Get the index with the highest probability
        yhat_idx = np.argmax(yhat)
        word = idx_to_word(yhat_idx, tokenizer)
        
        if word is None or word == 'endseq':
            break
            
        # --- REPETITION PENALTY ---
        # If the model predicts the same word twice in a row, take the 2nd best word
        words_so_far = in_text.split()
        if len(words_so_far) > 0 and word == words_so_far[-1]:
            yhat[0][yhat_idx] = 0 # Zero out the highest prob
            yhat_idx = np.argmax(yhat)
            word = idx_to_word(yhat_idx, tokenizer)
        
        in_text += ' ' + word
        
    return in_text.replace('startseq ', '').replace(' endseq', '')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if model and tokenizer:
        with st.spinner("🧠 Understanding the scene..."):
            # Preprocess image
            img_array = img_to_array(image)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            
            # Extract features
            feature = vgg_model.predict(img_array, verbose=0)
            
            # Generate Caption
            caption = predict_caption(model, feature, tokenizer)
            
            st.success("Analysis Complete!")
            st.subheader("Generated Caption:")
            st.write(f"### {caption.capitalize()}")
            
            # --- SPEECH OUTPUT ---
            try:
                tts = gTTS(text=caption, lang='en')
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp, format='audio/mp3', autoplay=True)
            except Exception as e:
                st.error("Speech generation failed, but the text is ready!")

    else:
        st.warning("⚠️ **Model Not Found**: Please check your file paths.")